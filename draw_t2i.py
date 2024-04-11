from clip_filter import clip_filter
from dalle_gen import dalle_gen, get_cls_index_name, description_refine, get_cls_template
import pandas as pd
from openai import OpenAI
import clip
import torch
import os


client = OpenAI(api_key='Replace with your own OPENAI KEY.')

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

thresh = 0.5
data_dir = 'datasets/Imagenet'


df = pd.read_csv('descriptions_data/extended_description.csv', header=None, names=['label', 'text'])
grouped_list = df.groupby('label')['text'].apply(list).to_dict()

for label, texts in grouped_list.items():
    index_name = get_cls_index_name(label)
    dir_path = os.path.join(data_dir, 'gen_train', str(index_name))
    os.makedirs(dir_path, exist_ok=True)
    
    for text_i in range(len(texts)):
        saved_path = os.path.join(dir_path, f"{index_name}_{text_i}.JPEG")

        img_path = dalle_gen(client, saved_path, texts[text_i], saved=True)
        if img_path != None:

            ## clip filter
            score = 0
            max_rounds = 2

            for round in range(max_rounds):
                if round > 0:
                    refine_texts = description_refine(texts[text_i], index_name)
                    saved_path_refine = os.path.join(dir_path, f"{index_name}_{text_i}_refine{round}.JPEG")
                    img_path = dalle_gen(client, saved_path_refine, refine_texts, saved=True)
                    if img_path != None:
                        score = clip_filter(img_path, cls_feature_template)
                else:
                    cls_feature_template = get_cls_template(index_name, label)
                    score = clip_filter(img_path, cls_feature_template)

                if score >= thresh:
                    break

