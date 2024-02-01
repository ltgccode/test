from openai import OpenAI
import pandas as pd
from data_txt.imagenet_label_mapping import get_readable_name
import csv

# api_key = "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"
# api_key = "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"
api_key = "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"

client = OpenAI(api_key=api_key)

df = pd.read_csv('descriptions_data/existing_description_list.csv', header=None, names=['label', 'text'])
grouped_texts = df.groupby('label')['text'].apply(lambda x: '\n'.join(x)).to_dict()
grouped_list = df.groupby('label')['text'].apply(list).to_dict()

# 打印每个标签对应的文本
for label, text in grouped_texts.items():
    current_all_description = grouped_list[label]
    # print(f"Label {label}:\n{text}\n")

    while len(current_all_description) < 20:
      real_name = get_readable_name(int(label)).split(", ")[0]

      system_content = "You will follow the Template to describe the object. Template: A photo of the class " + real_name + " {with distinctive features}{in specific scenes}. "
      current_description = text

      # print(current_description)

      user_content = "Besides these descriptions mentioned above, please use the same Template to list other possible {distinctive features} and {specific scenes} for the class " + real_name

      completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
          {"role": "system", "content": system_content},
          {"role": "user", "content": current_description},
          {"role": "user", "content": user_content}
        ]
      )

      output = completion.choices[0].message.content
      sentences = output.split(". ")
      current_all_description.append(sentences)

      with open('descriptions_data/extended_description.csv', mode='a', newline='') as file:
        writer = csv.writer(file)

        for s in sentences:
            writer.writerow([label, s])

      # print(completion.choices[0].message.content)