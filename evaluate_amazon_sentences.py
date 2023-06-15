from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from util import predict
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

# Load data
amazon_df = pd.read_csv('amazon_v0_sentences.csv')

# Make predictions
label_mapping = {'Human': 0, 'ChatGPT': 1, 'LABEL_0': 0, 'LABEL_1': 1}
amazon_df['pred'], amazon_df['pred_score'] = zip(*amazon_df['text'].progress_apply(
    predict,
    model_pipeline=model_pipeline, 
    tokenizer_kwargs=tokenizer_kwargs,
    label_mapping=label_mapping,

))

# Output to file
amazon_df.to_csv('amazon_v0_sentences_preds.csv')