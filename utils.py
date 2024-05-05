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


run_dir = f"drive/{getpass.getuser()}/project_data/calibrated_alignment/runs"


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
):
    """Load model from HuggingFace and wrap with PEFT if needed."""

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    # if "gen_kwargs" in hps:
    #    hps["gen_kwargs"]["pad_token_id"] = tokenizer.pad_token_id

    # Load model
    model_class = (
        transformers.AutoModelForSequenceClassification
        if reward_model
        else transformers.AutoModelForCausalLM
    )
    model = model_class.from_pretrained(model, torch_dtype=torch.bfloat16)
    model.to(torch.device("cuda:0"))
    model.config.pad_token_id = model.config.eos_token_id
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
        return sample

    # Load dataset
    dataset = datasets.load_dataset(name, data_dir)
    dataset = dataset["train"]
    # Split the dataset into training and testing subsets
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Make small if debug
    if debug:
        dataset["train"] = dataset["train"].select(range(10))
        dataset["test"] = dataset["test"].select(range(10))

    # Process dataset
    if name == "Dahoas/synthetic-instruct-gptj-pairwise":
        dataset = dataset.map(hh_rlhf_preprocess, batched=False)
        dataset = dataset.filter(lambda s: s["prompt"] is not None)
    elif name == "Dahoas/synthetic-instruct-gptj-pairwise":
        dataset = dataset.map(instruct_preprocess, batched=False)
        dataset = dataset.filter(lambda s: s["prompt"] is not None)
    return dataset
