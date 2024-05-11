from trl import SFTConfig, SFTTrainer
import utils
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLMWithValueHead,
)
import yaml
import wandb
import torch as t
from tqdm import tqdm


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

    # load model
    tokenizer, model = utils.load_model(
        hps["model"],
        reward_model=False,
        eval=False,
    )

    model = trl.AutoModelForCausalLMWithValueHead.from_pretrained(model)
    model.to(device)
    dataset = utils.load_dataset(tokenizer, **hps["dataset"], debug=False, sft=True)

    sft_config = SFTConfig(
        dataset_text_field="text",
        max_seq_length=1536,
        output_dir="/tmp",
    )

    trainer = SFTTrainer(
        hps["model"],
        train_dataset=dataset,
        args=sft_config,
    )
    trainer.train()
