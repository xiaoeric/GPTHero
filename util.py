from datasets import Dataset
import pandas as pd
import json
import nltk
nltk.download('punkt')
from nltk import sent_tokenize # for spliting English sentences
import torch

# Expand reviews into individual examples
def expand_reviews(raw_data, human='human', generated='chatgpt', from_pd=False):
  data = []
  for row in raw_data:
    if from_pd:
      _, row = row
    for human_review in row[human]:
      data.append({
          'text': human_review,
          'label': 0,
      })
    for generated_review in row[generated]:
      data.append({
          'text': generated_review,
          'label': 1,
      })
  return data

# Creates Trainer-ready dataset from data
def create_dataset(data):
  dataset = Dataset.from_pandas(pd.DataFrame(data=data))
  return dataset

# Opens .jsonl file
def open_jsonl(path):
  with open(path, 'r') as json_files:
    json_list = list(json_files)
  dicts = [json.loads(json_str) for json_str in json_list]
  return dicts

# Prepares HC3 data into a Huggingface Dataset
def prepare(data):
    answers = []
    for idx, example in enumerate(data):
        for human_answer in example["human_answers"]:
            answers.append({"label": 0,
                            "text": human_answer,
                            "data_idx": idx,})
        for gpt_answer in example["chatgpt_answers"]:
            answers.append({"label": 1, 
                            "text": gpt_answer,
                            "data_idx": idx,})
    return Dataset.from_list(answers)

# Splits human answers and chatgpt answers into individual entries with labels and source question indicies
def split_answers(data):
    answers = []
    for q_idx, row in enumerate(data.itertuples()):
        source = row.source
        for human_answer in row.human_answers:
            answers.append({"question_idx": q_idx,
                            "text": human_answer,
                            "label": 0,
                            "source": source})
        for gpt_answer in row.chatgpt_answers:
            answers.append({"question_idx": q_idx,
                            "text": gpt_answer,
                            "label": 1,
                            "source": source})
    return pd.DataFrame.from_records(answers)

# Splits answer data into individual sentences
def split(data):
    sentences = []
    for answer in data:
        for sentence in sent_tokenize(answer["text"]):
            sentences.append({"label": answer["label"],
                              "text": sentence,
                              "data_idx": answer["data_idx"],})
    return Dataset.from_list(sentences)

# Splits text data in a Pandas DataFrame into sentences
def split_df(df):
    sentences = []
    for row in df.itertuples():
        for sentence in sent_tokenize(row.text):
            sentences.append([
                row.label,
                sentence,
                row.data_idx,
            ])
    return pd.DataFrame(sentences, columns=['label', 'text', 'data_idx'])

# Performs model inference, used with Pandas apply
def predict(text, model_pipeline=None, tokenizer_kwargs={}, label_mapping=None):
    pred = model_pipeline(text, **tokenizer_kwargs)
    if label_mapping is not None:
      label = label_mapping[pred[0]['label']]
    else:
      label = pred[0]['label']
    return label, pred[0]['score']