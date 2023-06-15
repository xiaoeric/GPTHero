from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
import json
import datasets
import pandas as pd
from util import expand_reviews, create_dataset, predict
from tqdm import tqdm
tqdm.pandas()

# Define the path to model
MODEL_PATH = './model_amazon_v1'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0,
)
tokenizer_kwargs = {
    'padding':True,
    'truncation':True,
    'max_length':512,
}

# Import data
with open('amazon_v0.json', 'r') as f:
    data = json.load(f)
dataset = create_dataset(expand_reviews(data))

# Split the dataset into training and test sets
train_test = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
train_dataset = train_test['train']
test_dataset = train_test['test']

# Combine train and test datasets into Dataframe
amazon_train_df = train_dataset.to_pandas()
amazon_train_df['split'] = 'train'
amazon_test_df = test_dataset.to_pandas()
amazon_test_df['split'] = 'test'
amazon_df = pd.concat([amazon_train_df, amazon_test_df])

amazon_df['pred'], amazon_df['pred_score'] = zip(*amazon_df['text'].progress_apply(
    predict,
    model_pipeline=model_pipeline, 
    tokenizer_kwargs=tokenizer_kwargs
))

amazon_df.to_csv('amazon_v0_preds.csv')

# task_evaluator = evaluate.evaluator('text-classification')
# result = task_evaluator.compute(
#     model_or_pipeline=model,
#     tokenizer=tokenizer,
#     data=test_dataset,
#     metric=evaluate.combine(['accuracy','recall','precision','f1']),
#     label_mapping={'LABEL_0': 0, 'LABEL_1': 1},
# )

# print(result)