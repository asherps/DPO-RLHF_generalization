from trl import PPOConfig, PPOTrainer
import utils
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    BertModel,
    pipeline,
    AutoModelForSequenceClassification,
)
import yaml
import getpass
import wandb
from typing import Dict, Any
import torch as t
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import trl


device = t.device("cuda" if t.cuda.is_available() else "cpu")

<<<<<<< HEAD
#
def reward_fn(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    prompt_text: list[str],
    response_text: list[str],
    device: str,
) -> list[t.FloatTensor]:
    """Compute the reward for a given response to a prompt.

    Args:
        model (AutoModel): Huggingface model.
        tokenizer (AutoTokenizer): Huggingface tokenizer.
        prompt_text (list[str]): List of strings representing the prompt.
        response_text (list[str]): List of strings representing the response.
        device (str, optional): Device to run the model on. Defaults to 'cpu'.

    Returns:
        list[float]: A list of floats representing the reward.

    """
    with t.no_grad():
        encoding = tokenizer(
            prompt_text,
            response_text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt',
        )
        encoding = encoding.to(device)

        logits = model(**encoding).logits
        # scores = logits.cpu().numpy().flatten().tolist()

        return logits
=======
>>>>>>> 45e4f9b4aa1726566cfbba7db799fb87eb50101d

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
    def custom_collate(batch):
        input_ids = [item['input_ids'] for item in batch]
        queries = [item['query'] for item in batch]

        max_length = max(len(ids) for ids in input_ids)
        input_ids = [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]

        input_ids = t.tensor(input_ids)
        return {'input_ids': input_ids, 'queries': queries}

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
        reward_model=False,
        eval=False,
    )

    model = trl.AutoModelForCausalLMWithValueHead.from_pretrained(model)

    # Load and process dataset. Make eval set smaller for speed reasons.
    dataset = utils.load_dataset(tokenizer, **hps["dataset"], debug=True)
    test_size = min(len(dataset["test"]), 2_000)
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(test_size))
    # Setting logging
    logdir = setup_logging(hps)

    # I think PPO trainer fine tunes already, so we don't need this
#     peft_config = LoraConfig(
    
#     task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=16, lora_dropout=0.1,
# ) # create LoRA config for the finetuning

#     model = get_peft_model(model, peft_config) # create a model ready for LoRA finetuning

#     tokenizer.pad_token = tokenizer.eos_token # need this because tokenizer doesn't have default padding

#     # fine tune!
#     training_args = TrainingArguments(
#         output_dir="./results",
#         num_train_epochs=3,
#         per_device_train_batch_size=1,
#         per_device_eval_batch_size=2,
#         warmup_steps=500,
#         weight_decay=0.01,
#         logging_dir=logdir,
#         logging_steps=10,
#         learning_rate = 1e-3,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset,
#     )
#     trainer.train()

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
        sample["input_ids"] = tokenizer.encode(sample["query"])
        return sample

    dataset = dataset.rename_column("prompt", "query")
    dataset = dataset.map(tokenize, batched=False)
    dataset = dataset.remove_columns(["chosen", "rejected"])

    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        # dataset=dataset['train'],
        tokenizer=tokenizer,
    )

    dl = ppo_trainer.prepare_dataloader(dataset['train'], data_collator=custom_collate)

    # load reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained(hps["calibrated_model_path"])
    reward_model = reward_model.to(t.device("cuda:0")).eval()

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 10,
    }

    # wandb.init()
    print("Dataset size:", len(dataset['train']))

    epochs = 1
    for epoch in tqdm(range(epochs), "epoch: "):
        for batch in tqdm(dl):
            if batch is None:
                print('hi')
                continue
                
            query_tensors = batch["input_ids"]
            # print(query_tensors.shape)
            query_tensors = [tensor.view(-1) for tensor in query_tensors]
            #### Get response from SFTModel
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)

            batch["response"] = [
                tokenizer.decode(r.squeeze()) for r in response_tensors
            ]
            #### Compute reward score
            texts = [q + r for q, r in zip(batch["queries"], batch["response"])]
            chosen_scores = reward_fn(reward_model, tokenizer, batch["queries"], batch["response"], device)
            # rewards = [t.tensor(output[1]["score"]) for output in pipe_outputs]
            print(chosen_scores)
            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, chosen_scores)
            ppo_trainer.log_stats(stats, batch, rewards)

    #### Save model
    ppo_trainer.save_pretrained("my_ppo_model")


if __name__ == "__main__":
    main()