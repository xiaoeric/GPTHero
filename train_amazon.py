from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import json
import datasets
import pandas as pd
import wandb
import evaluate
from util import expand_reviews, create_dataset

# Define the model name and number of labels
MODEL_NAME = 'roberta-base'
NUM_LABELS = 2

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Import data
with open('amazon_v0.json', 'r') as f:
    data = json.load(f)

dataset = create_dataset(expand_reviews(data))

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Select the input and target columns
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Split the dataset into training and test sets
train_test = tokenized_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
train_dataset = train_test['train']
test_dataset = train_test['test']

# Initialize WandB
wandb.init(
    project='GPT-Hero', 
    entity='xiaoeric',
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    learning_rate=5e-5,
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    logging_strategy='steps',
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,
    report_to='wandb',
)

# Define the trainer
trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./saved_model/')
tokenizer.save_pretrained('./saved_model/')
# wandb.log_artifact(model)

# Evaluate the model on the test dataset
task_evaluator = evaluate.evaluator('text-classification')
result = task_evaluator.compute(
    model_or_pipeline=model,
    data=test_dataset,
    metric=evaluate.combine(['accuracy','recall','precision','f1']),
)

wandb.log(result)

# Finish the WandB run
wandb.finish()