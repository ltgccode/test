from clip_filter import clip_filter
from dalle_gen import dalle_gen
import pandas as pd
from openai import OpenAI
import clip
import torch


client = OpenAI(api_key='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')
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

saved_path = "dalle_gen/imagenet/white_siamese_cat_2.jpg"


df = pd.read_csv('descriptions_data/extended_description.csv', header=None, names=['label', 'text'])
grouped_list = df.groupby('label')['text'].apply(list).to_dict()

for label, texts in grouped_list.items():
    for text in texts:
        dalle_gen(client, saved_path, text, saved=False)
