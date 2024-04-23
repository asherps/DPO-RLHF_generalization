from typing import Any, Dict, Optional, Type

import getpass
import os
from xml.dom.expatbuilder import Rejecter
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import datasets
import peft
import torch
import transformers
import trl
import wandb
import calibrated_dpo
import utils


def setup_logging(hps: Dict[str, Any]):
    # Choose logging and checkpoint saving directory
    if hps["dataset"]["name"] == "Anthropic/hh-rlhf":
        hps["dataset_name"] = "hh_rlhf"
    logdir = utils.choose_log_dir(
        f"{utils.run_dir}/{hps['dataset_name']}/training/{hps['training_algorithm']}",
        debug=hps["debug"],
    )

    # Add a couple of keys to the hps object and save it as a yaml file
    hps["logdir"] = logdir
    hps["training_kwargs"]["run_name"] = "/".join(logdir.split("/")[-2:])
    hps["user"] = getpass.getuser()
    hps["tags"] += [
        hps["dataset"]["name"],
        "training",
        hps["training_algorithm"],
    ]
    with open(f"{logdir}/hps.yaml", "w") as f:
        yaml.dump(hps, f)

    # If not in debug mode, setup wandb logging
    if not hps["debug"]:
        wandb.init(
            project="Calibrated-Alignment",
            dir=logdir,
            name=hps["training_kwargs"]["run_name"],
            config=utils.wandb_configify(hps),
            tags=hps["tags"],
            save_code=True,
            settings=wandb.Settings(code_dir="."),
        )

    print(f"Hyperparameters:\n{hps}\n")
    return logdir


def build_trainer(
    tokenizer: transformers.AutoTokenizer,
    model: transformers.PreTrainedModel,
    dataset: datasets.DatasetDict,
    logdir: str,
    training_kwargs: Dict[str, Any],
    training_algorithm: str,
    training_algorithm_kwargs: Dict[str, Any],
    debug: bool,
    peft_config_class: Optional[Type[peft.PeftConfig]] = None,
    peft_config_kwargs: Optional[Dict[str, Any]] = None,
):
    # Build training args, which are independent of the specific training algorithm
    args_class = (
        trl.RewardConfig
        if training_algorithm == "reward_model"
        else transformers.TrainingArguments
    )
    train_args = args_class(
        output_dir=f"{logdir}/checkpoints",
        evaluation_strategy="steps",
        report_to="none" if debug else "wandb",
        remove_unused_columns=False,
        logging_steps=10,
        eval_steps=200,
        save_steps=0.2,
        **training_kwargs,
    )

    # Construct trainer
    if training_algorithm in ["dpo", "calibrated_dpo"]:
        trainer = calibrated_dpo.CalibratedDPOTrainer(
            model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            args=train_args,
            max_prompt_length=1024,
            max_length=1536,
            peft_config=peft_config_class(**peft_config_kwargs),
            **training_algorithm_kwargs,
        )
    elif training_algorithm == "reward_model":

        def prep_for_reward_trainer(sample):
            chosen = [p + c for p, c in zip(sample["prompt"], sample["chosen"])]
            chosen_inputs = tokenizer(
                chosen,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=1536,
            )
            rejected = [p + r for p, r in zip(sample["prompt"], sample["rejected"])]
            rejected_inputs = tokenizer(
                rejected,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=1536,
            )
            return {
                "input_ids_chosen": chosen_inputs["input_ids"],
                "attention_mask_chosen": chosen_inputs["attention_mask"],
                "input_ids_rejected": rejected_inputs["input_ids"],
                "attention_mask_rejected": rejected_inputs["attention_mask"],
            }

        train_args.max_length = 1536
        trainer = trl.RewardTrainer(
            model,
            train_dataset=dataset["train"].map(prep_for_reward_trainer, batched=True),
            eval_dataset=dataset["test"].map(prep_for_reward_trainer, batched=True),
            tokenizer=tokenizer,
            args=train_args,
            peft_config=peft_config_class(**peft_config_kwargs),
            **training_algorithm_kwargs,
        )
    return trainer


def main():
    # Load hyperparameters
    args = utils.argparser().parse_args()
    with open(
        args.hyperparam_file,
    ) as f:
        hps = yaml.load(f, Loader=yaml.FullLoader)
    # To keep debug runs short
    hps["debug"] = args.debug
    if hps["debug"]:
        hps["training_kwargs"]["max_steps"] = 5

    # Load model
    tokenizer, model = utils.load_model(
        hps["model"],
        reward_model=hps["training_algorithm"] == "reward_model",
        eval=False,
    )

    # Load and process dataset. Make eval set smaller for speed reasons.
    dataset = utils.load_dataset(tokenizer, **hps["dataset"], debug=args.debug)
    test_size = min(len(dataset["test"]), 2_000)
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(test_size))

    # Setting logging
    logdir = setup_logging(hps)

    # Train
    if hps["training_algorithm"] == "dpo":
        hps["training_algorithm_kwargs"]["alpha"] = hps["training_algorithm_kwargs"][
            "beta"
        ]
    trainer = build_trainer(
        tokenizer,
        model,
        dataset,
        logdir,
        hps["training_kwargs"],
        hps["training_algorithm"],
        hps["training_algorithm_kwargs"],
        peft_config_class=hps.get("peft_config_class"),
        peft_config_kwargs=hps.get("peft_config_kwargs"),
        debug=args.debug,
    )
    trainer.train()
    if not hps["debug"]:
        model.save_pretrained(f"{logdir}/checkpoints/final")
        wandb.finish()


if __name__ == "__main__":
    main()
