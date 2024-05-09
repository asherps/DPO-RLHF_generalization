from trl import PPOConfig, PPOTrainer
import utils
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForCausalLMWithValueHead,
    pipeline,
)
import yaml
import getpass
import wandb
from typing import Dict, Any
import torch as t
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
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

    # To keep debug runs short
    hps["debug"] = args.debug
    if hps["debug"]:
        hps["training_kwargs"]["max_steps"] = 5
    logdir = setup_logging(hps)

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model, torch_dtype=t.bfloat16
    )
    model.to(t.device(device))
    model.config.pad_token_id = model.config.eos_token_id

    # Load and process dataset. Make eval set smaller for speed reasons.
    dataset = utils.load_dataset(tokenizer, **hps["dataset"], debug=True)
    print(dataset)
    print(dataset.keys())
    breakpoint()
    test_size = min(len(dataset["test"]), 2_000)
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(test_size))

    config = PPOConfig(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        learning_rate=1.41e-5,
    )

    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
    }

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        return sample

    dataset = dataset.map(tokenize, batched=False)

    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        dataset=dataset["train"],
        tokenizer=tokenizer,
    )

    # load reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        hps["reward_model_path"]
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # wandb.init()

    epochs = 10
    for _ in tqdm(range(epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]

            #### Get response from SFTModel
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            batch["response"] = [
                tokenizer.decode(r.squeeze()) for r in response_tensors
            ]

            #### Compute reward score
            texts = [
                q + t + r
                for q, t, r in zip(batch["query"], "Assistant: ", batch["response"])
            ]
            pipe_outputs = reward_model(texts)
            print(texts[0])
            breakpoint()
            rewards = [t.tensor(output[1]["score"]) for output in pipe_outputs]

            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

    #### Save model
    ppo_trainer.save_pretrained("post_ppo_model")


if __name__ == "__main__":
    main()
