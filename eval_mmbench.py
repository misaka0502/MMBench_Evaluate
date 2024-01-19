import base64
import io
import random
import openai

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from test_main_new import Verifier
from lavis.lavis.models import load_model_and_preprocess

import torch
from tqdm import tqdm

import base64
import io
import random

import argparse
import os
from os import path as osp

import httpx
import time


import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
state = False

def decode_base64_to_image(base64_string, idx):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    global state
    if osp.exists(f"./mmbench_pics/img_{idx}.png"):
        state = False
        # print("state:False")
    elif not osp.exists(f"./mmbench_pics/img_{idx}.png"):
        image.save(f"./mmbench_pics/img_{idx}.png")
        state = True
        # print("state:True")
    return image

def get_result(template):
    result = pd.DataFrame(columns=["index", "question", "A", "B", "C", "D","answer", "prediction"])
    data = pd.read_csv(template, sep='\t')
    result["index"] = data["index"]
    result["question"] = data["question"]
    result["A"] = data["A"]
    result["B"] = data["B"]
    result["C"] = data["C"]
    result["D"] = data["D"]
    return result

class MMBenchDataset(Dataset):
    def __init__(self,
                 data_file,
                 vis_processors,
                 device,
                 sys_prompt='There are several options:'):
        self.df = pd.read_csv(data_file, sep='\t')
        self.sys_prompt = sys_prompt
        self.vis_processors = vis_processors
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        image = decode_base64_to_image(image, index)
        image = self.vis_processors["eval"](image).to(self.device)
        # print(f"Image shape after vis processor:{image.shape}")
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None
        catetory = self.df.iloc[idx]['category']
        l2_catetory = self.df.iloc[idx]['l2-category']

        option_candidate = ['A', 'B', 'C', 'D']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = f'{self.sys_prompt}\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        hint = self.load_from_df(idx, 'hint')
        if hint is not None:
            prompt = hint + ' ' + options_prompt + ' ' + question #+ '\nAnswer:'
        else:
            prompt = options_prompt + ' ' + question #+ '\nAnswer:'
        
        if answer != None:
            data = {
                "idx": index,
                'image': image,
                'question': prompt,
                'answer': answer,
            }
        else:
            data = {
                "idx": index,
                'image': image,
                'question': prompt,
            }
        return data
    
    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None

def main():
    #------------------------load model---------------------------------#
    device = torch.device("cuda:4") if torch.cuda.is_available() else "cpu"
    model_name = "instructblip_7b"
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_vicuna_instruct", 
        model_type="vicuna7b", 
        is_eval=True, 
        device=device)
    
    #------------------------load dataset------------------------------#
    dataset_name = "MMbench_TEST_EN"
    data_path = '/data/lindengtian/mmbench/test/mmbench_test_en_20231003.tsv'
    result_original = get_result(data_path)
    result_revise = get_result(data_path)
    dataset = MMBenchDataset(data_file=data_path, vis_processors=vis_processors, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # df = pd.read_csv(data_path, sep='\t')
    # image = df.iloc[1]['image']
    # image = decode_base64_to_image(image)
    # image = vis_processors["eval"](image).unsqueeze(0).to(device)
    # question = df.iloc[1]['question']
    # prompt = 'There are several options:' + ' ' + question
    # model.generate({"image": image, "prompt": prompt})

    #------------------------load verifier-------------------------------#
    verifier = Verifier(model, vis_processors, device)

    #------------------------evaluate---------------------------------#
    result_file_original = model_name + '_' + dataset_name
    result_file_revise = model_name + '_' + dataset_name + "revised"

    #------------统计当前已经验证过的数量，方便在程序意外终止后恢复-----------#
    file_list = os.listdir("mmbench_res/instructblip_7b/")
    file_cnt = len(file_list)
    #------------------------original model result---------------------#
    for id, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        if id >= (file_cnt*200):
            # image = sample["image"]
            # print(f"Image shape in samples:{image.shape}")
            #----------------------------评估原始答案-----------------------------#
            prompt = sample["question"][0] #+ "\nPlease output the options you selected in full"
            answer_original = model.generate({"image": sample["image"], "prompt": prompt})
            # print(f"type of prompt:{type(prompt)}")
            # print(f"orignal prompt:{prompt}")
            # print(f"original answer:{answer_original}")
            for idx, out in zip(sample["idx"], answer_original):
                result_original.loc[result_original["index"].isin([idx.item()]),"prediction"] = out
                # print(f"out 1:{out}")
            if id % 200 == 0 and id != 0:
                result_original.to_excel(f"./mmbench_res/{model_name}/{result_file_original}_{id}.xlsx")

            #----------------------------评估验证后的答案--------------------------#
            try:
                answer_revise = verifier.verify(original_image=sample["image"], original_q=prompt, original_a=answer_original[0])
            except openai.InternalServerError:
                time.sleep(300)
                print("HTTPStatusError,waiting...")
                try:
                    answer_revise = verifier.verify(original_image=sample["image"], original_q=prompt, original_a=answer_original[0])
                except openai.InternalServerError:
                    print(f"Failed to verify:index:{idx}")
                    continue
            # print(f"revised answer 1:{answer_revise}")
            answer_revise = answer_revise.replace("A.","").replace("B.","").replace("C.","").replace("D.","")
            # print(f"revised answer 2:{answer_revise}")
            for idx, out in zip(sample["idx"], [answer_revise]):
                result_revise.loc[result_revise["index"].isin([idx.item()]),"prediction"] = out
                # print(f"out 2:{out}")
            if id % 200 == 0 and id != 0:
                result_revise.to_excel(f"./mmbench_res/{model_name}/{result_file_revise}_{id}.xlsx")
            # print("running")
        else:
            continue
    result_original.to_excel(f"./mmbench_res/{model_name}/{result_file_original}.xlsx")
    print(f"./mmbench_res/{model_name}/{result_file_original}.xlsx")
    result_revise.to_excel(f"./mmbench_res/{model_name}/{result_file_revise}.xlsx")
    print(f"./mmbench_res/{model_name}/{result_file_revise}.xlsx")
    # print(type(dataloader))
    # for batch_data in dataloader:
    #     # print(batch_data["question"])
    #     pass
if __name__ == '__main__':
    main()