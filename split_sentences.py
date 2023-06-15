import json
from util import expand_reviews, create_dataset, split_df

# Import data
with open('amazon_v0.json', 'r') as f:
    data = json.load(f)

dataset = create_dataset(expand_reviews(data))
df = dataset.to_pandas()
df.to_csv('amazon_v0.csv')
df['data_idx'] = df.iloc
sentences_df = split_df(df)
sentences_df.to_csv('amazon_v0_sentences.csv')