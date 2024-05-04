import random
import pandas as pd
from operator import itemgetter
import torch
import warnings
warnings.filterwarnings('ignore')
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification,AutoTokenizer,TrainingArguments
from trl import RewardTrainer

from datasets import load_dataset

dataset = load_dataset("truthful_qa")

df = pd.read_csv('feedback.csv')
df.head()

df['tup'] = list(zip(df['answer'], df['feedback']))
#grouping together all the answers for a given question along with its feedback
df_g = df.groupby('question')['tup'].apply(list).reset_index()
# sort each group based on the feedback score
df_g["sorted_tup"] = df_g["tup"].apply(lambda x :sorted(x,key=itemgetter(1)) )
# answer with highest feedback score is "chosen"
df_g["chosen"] = df_g["sorted_tup"].apply(lambda x: x[-1][0])
df_g["chosen_score"] = df_g["sorted_tup"].apply(lambda x: x[-1][1])
# answer with highest feedback score is "rejected"
df_g["rejected"] = df_g["sorted_tup"].apply(lambda x: x[0][0])
df_g["rejected_score"] = df_g["sorted_tup"].apply(lambda x: x[0][1])
df_g = df_g.dropna()
df_g = df_g[(df_g['chosen_score']>=4.0) & (df_g['rejected_score']<4.0)]

rows = []
for record in df_g.itertuples(index=True, name='Pandas'):
    if record is None or len(record) == 0:
        continue
    rows.append({
        "instruction": record.question,
        "chosen_response": record.chosen,
        "rejected_response": record.rejected
    })


prepared_dataset = Dataset.from_list(rows)
prepared_dataset.to_pandas()

# using mistral 
model_name = "facebook/m2m100_418M"
#Select a base model whch we need to train for reward modeling.
# model_name = "distilroberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

def formatting_func(examples):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}
    prompt_plus_chosen_response = examples["instruction"] + "\n" + examples["chosen_response"]
    prompt_plus_rejected_response = examples["instruction"] + "\n" + examples["rejected_response"]
    tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
    tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
    }
formatted_dataset = prepared_dataset.map(formatting_func)
formatted_dataset = formatted_dataset.train_test_split()
# Configuring the training arguments
training_args = TrainingArguments(
    output_dir="./reward_model",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    logging_steps=1,
    num_train_epochs = 10,
    report_to=None,
)
# Loading the RewardTrainer from TRL
trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
)
trainer.train()
# Saving the model
trainer.save_model("reward_model")
