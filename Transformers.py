
# Importing necessary libraries
#install the necessary libraries
# change this script accordingly while training gpt2, distilgpt2
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline

import math
import torch
from tqdm import tqdm
from datasets import load_metric
from evaluate import load

# Global variables and initializations
MODEL ='distilgpt2' #change to 'gpt2'
TRAIN_PATH =   "/data/train_df.csv"
VALID_PATH =   "/data/valid_df.csv"
TEST_PATH =   "/data/test_df.csv"
TRAIN_ROWS = 50000
TEST_ROWS = 5000
CONTEXT_LEN = 256
TRAIN_BS = 64
TEST_BS = 64
EPOCHS = 5


def prepare_data():
    dataset = load_dataset("csv", data_files={"train": TRAIN_PATH, "test": VALID_PATH})
    dataset['train'] = dataset['train'].select(range(TRAIN_ROWS))
    dataset['test'] = dataset['test'].select(range(TEST_ROWS))
    return dataset


def tokenize(element):
    tokenizer = AutoTokenizer.from_pretrained(distilgpt)
    outputs = tokenizer(
        element["stories"],
        truncation=True,
        max_length=CONTEXT_LEN,
        return_overflowing_tokens=True,
        return_length=True,
    )
    
    input_batch = []
    
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == CONTEXT_LEN:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}
    
def data_preprocess(dataset):
    dataset = dataset.remove_columns(['Unnamed: 0'])
    tokenized_datasets = dataset.map(
    tokenize, batched=True, remove_columns=dataset["train"].column_names)
    return tokenized_datasets

def define_model():
    model = AutoModelForCausalLM.from_pretrained(distilgpt)
    return model

def train_and_save_model():
    # Training related code here
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataset = prepare_data()
    tokenized_datasets = data_preprocess(dataset)
    model = define_model()
    
    args = TrainingArguments(
        output_dir="ai-story_gen",
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=TEST_BS,
        evaluation_strategy="steps",
        eval_steps=1_50,
        logging_steps=1_50,
        gradient_accumulation_steps=8,
        num_train_epochs=EPOCHS,
        weight_decay=0.1,
        warmup_steps=1_00,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        fp16=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )
    trainer.train()

    trainer.save_model("./story_gpt2_finetune")

def evaluate_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = pipeline(
    "text-generation", model=model,tokenizer=tokenizer, device=device
    )
    txt = "Once upon a time, there lived a Lion"
print(pipe(txt, num_return_sequences=1,max_new_tokens=100,
          pad_token_id=tokenizer.eos_token_id)[0]["generated_text"])


# Main function
def main():
    train_and_save_model()
    evaluate_model()

# Entry point
if __name__ == "__main__":
    main()
