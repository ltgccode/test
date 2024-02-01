import torch
# from torchvision import transforms
from lt_dataloaders import ImageNetLTDataLoader
from data_txt.imagenet_label_mapping import get_readable_name
from gpt4v import gpt4v_observe
from ultis import sample_counter
import os
import json
import csv


# 数据目录和其他参数
data_dir = 'Imagenet'
batch_size = 1 # Not for training! Cannot change!
num_workers = 4
training = True

imagenet_loader = ImageNetLTDataLoader(data_dir=data_dir, 
                                       batch_size=batch_size, 
                                       shuffle=False, 
                                       num_workers=num_workers, 
                                       training=training)

class_number_file = 'data_txt/ImageNet_LT/imagenetlt_class_count.txt'
if not os.path.exists(class_number_file):
    sample_counter(imagenet_loader)
    with open(class_number_file, 'r') as file:
        dict_class_number = json.load(file)
else:
    with open(class_number_file, 'r') as file:
        dict_class_number = json.load(file)
data_to_write = []

for epoch, pack in enumerate(imagenet_loader):
    data, target, index = pack

    if dict_class_number[str(int(target))] < 25:
        real_name = get_readable_name(int(target)).split(", ")[0]
        text_prompt="Template: A photo of the class "+real_name+", {with distinctive features}{in specific scenes}. Please use the Template to briefly describe the image of the class " + real_name + '.'
        # print(text_prompt)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        data = data * std + mean

        # img_description = gpt4v_observe(data, text_prompt)['choices'][0]['message']['content']
        img_description = gpt4v_observe(data, text_prompt)

        # print(img_description)
        
        if img_description[0] == 'A':
            data_to_write.append((int(target), img_description))

            if epoch % 5 == 0:
                with open('descriptions_data/existing_description_list.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(data_to_write)
                data_to_write = []