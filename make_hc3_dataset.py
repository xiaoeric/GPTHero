from transformers import pipeline
from util import open_jsonl, prepare, split
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

hc3_all = open_jsonl('HC3_all.jsonl')
hc3_all_dataset = prepare(hc3_all)
hc3_all_sentences = split(hc3_all_dataset)

hc3_all_df = hc3_all_sentences.to_pandas()
hc3_all_df.to_csv('hc3_all_sentences.csv')