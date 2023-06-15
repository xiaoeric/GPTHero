from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import json
import datasets
import pandas as pd

# Define the model name and number of labels
MODEL_NAME = "roberta-base"
NUM_LABELS = 2

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Import data
with open('amazon_v0.json', 'r') as f:
    data = json.load(f)

# Expand reviews into individual examples
def expand_reviews(raw_data, human='human', generated='chatgpt', from_pd=False):
  review_data = []
  for row in raw_data:
    if from_pd:
      _, row = row
    for human_review in row[human]:
      review_data.append({
          'text': human_review,
          'label': 0,
      })
    for generated_review in row[generated]:
      review_data.append({
          'text': generated_review,
          'label': 1,
      })
  return review_data

# Creates Trainer-ready dataset from data
def create_dataset(data):
  dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
  return dataset

dataset = create_dataset(expand_reviews(data))
# TODO remove later, train on small sample to test code
dataset = dataset.select(range(10))

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

print(train_dataset)