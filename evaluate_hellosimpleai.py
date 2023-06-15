from transformers import pipeline
# from util import open_jsonl, prepare, split
# from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# hc3_all = open_jsonl('HC3_all.jsonl')
# hc3_all_dataset = prepare(hc3_all)
# hc3_all_sentences = split(hc3_all_dataset)

label_mapping = {'Human': 0, 'ChatGPT': 1}
model_pipeline = pipeline(
    "text-classification", 
    model="Hello-SimpleAI/chatgpt-detector-roberta", 
    tokenizer="Hello-SimpleAI/chatgpt-detector-roberta",
    device=0,
)
tokenizer_kwargs = {
    'padding':True,
    'truncation':True,
    'max_length':512,
}
def predict(text):
    pred = model_pipeline(text, **tokenizer_kwargs)
    return pred[0]['label'], pred[0]['score']

hc3_all_df = pd.read_csv('hc3_all_sentences.csv')  # hc3_all_sentences.to_pandas()
# hc3_all_df = hc3_all_df.iloc[219010:219014]  # hc3_all_df.sample(frac=0.1, random_state=42)
# for i in range(len(hc3_all_df)):
#     text = hc3_all_df.iloc[i]['text']
#     print(f'{i}:', text)
#     print(f'pred for {i}:', predict(text))

hc3_all_df['pred'], hc3_all_df['pred_score'] = zip(*hc3_all_df['text'].progress_map(predict))

hc3_all_df.to_csv("hc3_all_sentences_preds.csv")