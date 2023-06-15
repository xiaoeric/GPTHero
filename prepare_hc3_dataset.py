from util import open_jsonl, split_answers
import pandas as pd

hc3_all = open_jsonl('./data/HC3_all.jsonl')
hc3_all_df = pd.DataFrame.from_records(hc3_all)
hc3_all_df.to_csv('./data/hc3_all.csv', index=False)

hc3_all_dataset_df = split_answers(hc3_all_df)
hc3_all_dataset_df.to_csv('./data/hc3_all_dataset.csv', index=False)