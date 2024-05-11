from trl import SFTTrainer, AutoModelForCausalLMWithValueHead
import utils
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments
)
import yaml
import getpass
import wandb
import torch as t
from tqdm import tqdm
from typing import Dict, Any


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
    logdir = setup_logging(hps)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=t.bfloat16
    )

    # load model
    tokenizer, model = utils.load_model(
        hps["model"],
        reward_model=False,
        eval=False,
        quantized =True, 
        bnb_config=bnb_config,
    )
    dataset = utils.load_dataset(tokenizer, **hps["dataset"], debug=False, sft=True)

    def memory():
        allocated_memory = t.cuda.memory_allocated()
        print(f"memory allocated: {allocated_memory / (2**30)}")

    print(memory())

    args = TrainingArguments(
    output_dir="sft_model", # directory to save and repository id
    num_train_epochs=1,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
)
    trainer = SFTTrainer(
        model,
        tokenizer= tokenizer,
        train_dataset=dataset["train"].select(range(40000)),
        args = args,
        dataset_text_field="prompt",
        dataset_batch_size = 1000,
        eval_dataset = dataset["test"], 
        peft_config=hps["peft_config_class"](**hps["peft_config_kwargs"]),
        packing = True, 
        max_seq_length = 3072
        )
    trainer.train()

    trainer.save_model()
    wandb.finish()
    del model
    del trainer
    torch.cuda.empty_cache()
if __name__ == "__main__":
    main()
