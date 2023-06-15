import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import wandb
import evaluate
from util import expand_reviews, create_dataset
from torch.utils.data import DataLoader

# Check for device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print ('GPU will be used')
else:
    print ("CPU will be used")

# Params
params = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 6,
    'max_epochs': 100,
}

# Define the model name and number of labels
MODEL_NAME = 'roberta-base'
NUM_LABELS = 2

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(device)

# Import data
hc3_all_df = pd.read_csv("./data/hc3_all_dataset.csv")

domain_label_map = {
    'reddit_eli5': 0,
    'open_qa': 1,
    'wiki_csai': 2,
    'finance': 3,
    'medicine': 4,
}

# Obtain pretrained xlm_r representations
def xlm_r_train(data: pd.DataFrame, tokenizer, domain_label_map):
    embedded_data = []
    for row in data.itertuples():
        encoded = torch.LongTensor(tokenizer.encode(row.text, truncation=True, padding='max_length')).to(device)
        domain_label = 

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