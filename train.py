from typing import Any, Dict, Optional, Type, Union, Tuple, Literal, List
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    PreTrainedModel,
)
import getpass
import datasets
import peft
import transformers
import gc
import trl
import wandb
import utils

""" THIS FILE TRAINS EITHER DPO OR A REWARD MODEL GIVEN A DATASET -- Depending 
 on which hyperparams you pass it. For instance to train DPO, run python train.py hyperparams/dpo.yaml"""


class DPOTrainer(trl.DPOTrainer):
    _tag_names = ["trl"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        alpha: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            model=model,
            ref_model=ref_model,
            **kwargs,
        )
        self.alpha = alpha

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)

        logits = self.beta * pi_logratios - self.beta * ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            )

        chosen_rewards = (
            self.beta * policy_chosen_logps.to(self.accelerator.device)
            - self.beta * reference_chosen_logps.to(self.accelerator.device)
        ).detach()
        rejected_rewards = (
            self.beta * policy_rejected_logps.to(self.accelerator.device)
            - self.beta * reference_rejected_logps.to(self.accelerator.device)
        ).detach()

        # Here we renormalize logps such that p(chosen) + p(rejected) = 1
        # We then use this renormalized distribution to compute pairwise ce, entropy, and kl.
        policy_logps = torch.stack([policy_chosen_logps, policy_rejected_logps], dim=0)
        log_c_policy = -torch.logsumexp(policy_logps, dim=0)
        renormed_policy_logps = policy_logps - log_c_policy

        reference_logps = torch.stack(
            [reference_chosen_logps, reference_rejected_logps], dim=0
        )
        log_c_reference = -torch.logsumexp(reference_logps, dim=0)
        renormed_reference_logps = reference_logps - log_c_reference

        ce = -(renormed_policy_logps.exp() * renormed_reference_logps).sum(dim=0)
        entropy = -(renormed_policy_logps.exp() * renormed_policy_logps).sum(dim=0)
        kl = ce - entropy

        return losses, {
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
            "pairwise_cross_entropy": ce,
            "pairwise_entropy": entropy,
            "pairwise_kl": kl,
        }

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, metrics_dict = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (
            metrics_dict["chosen_rewards"] > metrics_dict["rejected_rewards"]
        ).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = metrics_dict["chosen_rewards"].mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = (
            metrics_dict["rejected_rewards"].mean().cpu()
        )
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (
            (metrics_dict["chosen_rewards"] - metrics_dict["rejected_rewards"])
            .mean()
            .cpu()
        )
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = (
            policy_rejected_logits.detach().mean().cpu()
        )
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        metrics[f"{prefix}pairwise_cross_entropy"] = (
            metrics_dict["pairwise_cross_entropy"].mean().cpu()
        )
        metrics[f"{prefix}pairwise_entropy"] = (
            metrics_dict["pairwise_entropy"].mean().cpu()
        )
        metrics[f"{prefix}pairwise_kl"] = metrics_dict["pairwise_kl"].mean().cpu()

        return losses.mean(), metrics


def setup_logging(hps: Dict[str, Any]):
    # Choose logging and checkpoint saving directory
    if hps["dataset"]["name"] == "UCL-DARK/sequential-instructions":
        hps["dataset_name"] = "sequential-instructions"
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
            project="dpo_rlhf_generalization",
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
    if training_algorithm in ["dpo"]:
        trainer = DPOTrainer(
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
            chosen = [p + c for p, c in zip(sample["instruction"], sample["output"])]
            
            chosen_inputs = tokenizer(
                chosen,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=1536,
            )
            rejected = [p + r for p, r in zip(sample["instruction"], sample["output"])]
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

    tokenizer, model = utils.load_model(
        hps["model"],
        reward_model=hps["training_algorithm"] == "reward_model",
        eval=False,
    )

    # Load and process dataset. Make eval set smaller for speed reasons.
    dataset = utils.load_dataset(tokenizer, **hps["dataset"], debug=args.debug)
    dataset["test"] = dataset["train"].select(range(483, 533))
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
