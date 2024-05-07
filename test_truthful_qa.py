import random
import torch
from typing import Optional
import yaml
import datasets
import transformers
import tqdm
import utils



def load_model(
    checkpoint_path: str,
):
    """Load model from a checkpoint and wrap with PEFT if needed."""

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model from the specific checkpoint
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16
    ).to(torch.device("cuda:0"))

    return tokenizer, model


def load_dataset(
    name: str,  # HuggingFace name of dataset
    data_dir: Optional[str] = None,  # Specifies splits of dataset to use
    # debug = False,
):
    """Load and preprocess dataset."""

    dataset = datasets.load_dataset(name, data_dir)
    return dataset


def shuffle_lists_in_unison(list1, list2):
    zipped_lists = list(zip(list1, list2))
    random.shuffle(zipped_lists)
    list1_shuffled, list2_shuffled = zip(*zipped_lists)
    return list(list1_shuffled), list(list2_shuffled)


def preprocess_truthful_qa(dataset, tokenizer):
    questions = dataset["question"]
    answers = dataset["mc1_targets"]
    final_labels = []  # preprocessed labels
    english_questions = []  # non-tokenized questions
    for answer, question in zip(answers, questions):
        # create formatted question
        choices = answer["choices"][:4]
        labels = answer["labels"][:4]
        rand_choices, rand_labels = shuffle_lists_in_unison(choices, labels)
        if len(rand_choices) < 4:
            continue
        formatted_question = f"""Please answer the following multiple choice question. Start your answer with the capital letter corresponding to your choice:
{question}

Answer choices: 
A. {rand_choices[0]} 
B. {rand_choices[1]}
C. {rand_choices[2]}
D. {rand_choices[3]}

Your answer: """

        formatted_question = tokenizer.apply_chat_template(
            [{"role": "user", "content": formatted_question}], tokenize=False
        )

        final_labels.append(rand_labels)
        english_questions.append(formatted_question)

    return final_labels, english_questions


def model_answers():
    # Load hyperparameters and setup
    args = utils.argparser().parse_args()
    with open(args.hyperparam_file) as f:
        hps = yaml.load(f, Loader=yaml.FullLoader)

    # Load models and tokenizer
    # tokenizer, _ = load_model(hps["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer = transformers.AutoTokenizer.from_pretrained(hps["model_path"])
    # tokenizer, model = load_model("mistralai/Mistral-7B-Instruct-v0.2")
    _, model = load_model("./drive/user/project_data/calibrated_alignment/runs/instruct/training/dpo/run_1/checkpoints/checkpoint-10000")
    model = model.to(torch.device("cuda:0")).eval()

    # Load and preprocess the dataset
    dataset = load_dataset( # tokenizer,
        "truthful_qa", "multiple_choice", # debug=args.debug
    )["validation"]
    subset = dataset[: min(1000, len(dataset))]
    labels, english_questions = preprocess_truthful_qa(subset, tokenizer)

    # find IDs for ABCD tokens
    tokenized_abcd = tokenizer(["A", "B", "C", "D"])
    abcd_ids = [ids[1] for ids in tokenized_abcd["input_ids"]]

    # Process the data in batches
    batch_size = 1
    results = []
    for i in tqdm.trange(0, len(english_questions), batch_size):

        batch_questions = tokenizer(
            english_questions[i : i + batch_size],
            return_tensors="pt",
            # padding="max_length",
            # truncation=True,
            # max_length=256,
            padding=False,
        ).to(model.device)

        batch_labels = labels[i : i + batch_size]
        eng_qs = english_questions[i : i + batch_size]

        # Run the model on the batch
        with torch.no_grad():
            batch_result = model(**batch_questions)
            abcd_logits = batch_result.logits[0, -1, abcd_ids].cpu()
            abcd_probs = torch.softmax(abcd_logits, dim=-1)
            info = {}
            info["abcd_probs"] = abcd_probs  # probs for regular model
            info["question"] = eng_qs  # question + answer choices (prompt)
            info["labels"] = batch_labels  # labels for each answer choice
            results.append(info)

    return results


def accuracy(results):
    """finds a model's accuracy on truthfulQA"""
    choices = torch.stack([result["abcd_probs"] for result in results])
    labels = torch.tensor([result["labels"] for result in results]).squeeze(dim=1)
    return (
        (torch.argmax(choices, dim=-1) == torch.argmax(labels, dim=-1)).float().mean()
    )


if __name__ == "__main__":
    results = model_answers()
    accuracy = accuracy(results)
    print(f"Accuracy: {accuracy}")

    logdir = utils.choose_log_dir(
        f"{utils.run_dir}/truthful_qa/calibration_metrics", debug=False
    )
    torch.save(results, f"{logdir}/results.pth")
