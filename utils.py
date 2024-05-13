from typing import Any, Dict, Optional, Type

import argparse
import getpass
import glob
import os
import yaml

import datasets
import peft
import torch
import transformers
import trl


def check_cuda_gpu_availability():
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"Using CUDA GPU: {device}")
    else:
        print("CUDA GPU is not available.")


run_dir = f"data"


def argparser(include_hyperparams: bool = True):
    """Utility function to create an argument parser with a couple default args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    if include_hyperparams:
        parser.add_argument("hyperparam_file")  # H`YPERPARAM FILE
    return parser


def choose_log_dir(base_dir: str, debug: bool = False, make_dir: bool = True):
    """
    Utility function to choose a log directory for a run.
    The log directory is base_dir/run_{max_previous_run_number + 1}.
    """
    if debug:
        base_dir += "/debug"
    max_run_number = max(
        [0] + [int(r.split("_")[-1]) for r in glob.glob(f"{base_dir}/run_*")]
    )
    logdir = f"{base_dir}/run_{max_run_number + 1}"
    if make_dir:
        os.makedirs(logdir, exist_ok=True)
    return logdir


def wandb_configify(val: Any):
    """Utility method to turn a value into something that can be logged to wandb."""
    if isinstance(val, list):
        return [wandb_configify(v) for v in val]
    if isinstance(val, dict):
        return {k: wandb_configify(v) for k, v in val.items()}
    if hasattr(val, "__dict__"):
        return yaml.dump(val)
    return val


def load_model(
    model: str,
    reward_model: bool = False,
    eval: bool = True,
    PPO: bool = False,
    quantized: bool = False,
    bnb_config: dict = False,
):
    """Load model from HuggingFace and wrap with PEFT if needed."""

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, padding_side="left")
    tokenizer.pad_token = '$'
    # if "gen_kwargs" in hps:
    #    hps["gen_kwargs"]["pad_token_id"] = tokenizer.pad_token_id

    # Load model
    model_class = (
        transformers.AutoModelForSequenceClassification
        if reward_model
        else transformers.AutoModelForCausalLM
    )

    # model_class = (
    #     transformers.AutoModelForCausalLMWithValueHead
    #     if PPO else model_class
    # )
    if reward_model:
        model = model_class.from_pretrained(
            model, torch_dtype=torch.bfloat16, num_labels=1
        ).to(torch.device("cuda:0"))
    elif quantized:
        assert bnb_config is not None
        model = model_class.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            # load_in_4bit=True,
            quantization_config=bnb_config,
        )
    else:
        model = model_class.from_pretrained(model, torch_dtype=torch.bfloat16).to(
            torch.device("cuda:0")
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    if eval:
        model.eval()

    return tokenizer, model


def load_generation_config(
    model,
    generation_kwargs: Dict[str, Any],
):
    if "mistral" not in model.config.name_or_path:
        raise NotImplementedError(
            "Only works for Mistral models. Other models can have default generation configs that mess with our evaluation."
        )

    # Some default generation_kwargs
    generation_kwargs["do_sample"] = generation_kwargs.pop("do_sample", True)
    generation_kwargs["max_new_tokens"] = generation_kwargs.pop("max_new_tokens", 512)
    generation_kwargs["top_k"] = generation_kwargs.pop("top_k", model.config.vocab_size)

    # Create generation config
    if model.generation_config is None:
        base_gen_config = transformers.GenerationConfig()
    else:
        base_gen_config = transformers.GenerationConfig.from_model_config(
            model.generation_config
        )

    generation_config = transformers.GenerationConfig(
        **(base_gen_config.to_dict() | generation_kwargs)
    )
    return generation_config


def load_dataset(
    tokenizer: transformers.AutoTokenizer,
    name: str,  # HuggingFace name of dataset
    data_dir: Optional[str],  # Specifies splits of dataset to use
    debug: bool,
    sft=False,
) -> datasets.DatasetDict:
    """Load and preprocess dataset."""

    def hh_rlhf_preprocess(sample):
        # Process into conversation
        text = sample["chosen"]
        human_idx = 0
        human_tag = "\n\nHuman: "
        assistant_tag = "\n\nAssistant: "
        messages = []

        while True:
            try:
                assistant_idx = text.index("\n\nAssistant: ", human_idx)
                messages.append(
                    {
                        "role": "user",
                        "content": text[human_idx + len(human_tag) : assistant_idx],
                    }
                )
                next_human_idx = text.find(human_tag, assistant_idx)
            except ValueError as e:
                break
            if next_human_idx == -1:
                messages.append(
                    {
                        "role": "assistant",
                        "content": text[assistant_idx + len(assistant_tag) :],
                    }
                )
                break
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": text[
                            assistant_idx + len(assistant_tag) : next_human_idx
                        ],
                    }
                )
                human_idx = next_human_idx

        # Grab base conversation vs final completions
        sample["prompt"] = tokenizer.apply_chat_template(messages[:-1], tokenize=False)
        sample["chosen"] = sample["chosen"][assistant_idx + 13 :]
        sample["rejected"] = sample["rejected"][assistant_idx + 13 :]
        if sft:
            sample["prompt"] = sample["prompt"] + sample["chosen"]
        return sample

    def instruct_preprocess(sample):
        messages = []
        messages.append(
            {
                "role": "user",
                "content": sample["prompt"],
            }
        )

        # Grab base conversation vs final completions
        sample["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False)
        sample["chosen"] = sample["chosen"]
        sample["rejected"] = sample["rejected"]
        if sft:
            sample["prompt"] = sample["prompt"] + sample["chosen"]
        return sample

    # Load dataset
    dataset = datasets.load_dataset(name, data_dir)
    dataset = dataset["train"]
    # Split the dataset into training and testing subsets
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Make small if debug
    if debug:
        dataset["train"] = dataset["train"].select(range(1000))
        dataset["test"] = dataset["test"].select(range(1000))
    # Process dataset
    if name == "Anthropic/hh-rlhf":
        dataset = dataset.map(hh_rlhf_preprocess, batched=False)
        # dataset = dataset.filter(lambda s: s["prompt"] is not None)

    elif name == "Dahoas/synthetic-instruct-gptj-pairwise":
        dataset = dataset.map(instruct_preprocess, batched=False)
        dataset = dataset.filter(lambda s: s["prompt"] is not None)
    elif name == "Unified-Language-Model-Alignment/Anthropic_HH_Golden":
        dataset = dataset.map(hh_rlhf_preprocess, batched=False)
        dataset = dataset.filter(lambda s: s["prompt"] is not None)

    return dataset
