from trl import PPOConfig
import utils
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BertModel, pipeline
import yaml
import getpass
import wandb
from typing import Dict, Any
import torch as t
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


device = t.device("cuda" if t.cuda.is_available() else "cpu")


def setup_logging(hps: Dict[str, Any]):
    # Choose logging and checkpoint saving directory
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

    # load model
    tokenizer, model = utils.load_model(
        hps["model"],
        reward_model=hps["training_algorithm"] == "reward_model",
        eval=False,
    )

    # Load and process dataset. Make eval set smaller for speed reasons.
    dataset = utils.load_dataset(tokenizer, **hps["dataset"], debug=args.debug)
    test_size = min(len(dataset["test"]), 2_000)
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(test_size))


    config = PPOConfig(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        learning_rate=1.41e-5,
    )

    # Setting logging
    logdir = setup_logging(hps)
    

    peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=16, lora_dropout=0.1,
) # create LoRA config for the finetuning

    model = get_peft_model(model, peft_config) # create a model ready for LoRA finetuning

    tokenizer.pad_token = tokenizer.eos_token # need this because tokenizer doesn't have default padding

    # fine tune!
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logdir,
        logging_steps=10,
        learning_rate = 1e-3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()



if __name__ == "__main__":
    main()
