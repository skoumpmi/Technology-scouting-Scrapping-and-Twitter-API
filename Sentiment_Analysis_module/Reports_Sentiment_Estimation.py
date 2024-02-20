import numpy as np
import pandas as pd
import datetime
import os
from datetime import timedelta

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

import sys
sys.path.insert(1, './BERT')

from BERT.bert_classifiers import linearClassifier
from BERT.tools import InputExample, BinaryClassificationProcessor
from BERT.convert_examples_to_features import convert_example_to_feature

from transformers import BertTokenizer
from multiprocessing import Pool, cpu_count
from contextlib import closing

from tqdm import tqdm

from scipy.special import softmax
from scipy.stats import norm, t


from utilities.generalUtils import GeneralUtils
import pandas as pd
import os
import random as rd
from os import listdir
from os.path import isfile, join
import glob
import json




decay_past_window = 90
def generate_eval_features(eval_examples_for_processing, eval_examples_len):
    
        process_count = cpu_count() - 1 
        eval_features = list(tqdm(map(convert_example_to_feature, \
                                            eval_examples_for_processing), total=eval_examples_len))
           
        return eval_features
def predict_sentiment(text, path_to_model_directory):
        """ predicts the sentiment of the text using the finetuned_model, after
            tokenizing the text with the given tokenizer """
        """ select gpu or cpu"""   
        device = "cpu"        
        max_seq_length =  128 
        eval_batch_size = 8      
        processor = BinaryClassificationProcessor()      
        eval_examples = [InputExample(guid=0, text_a=text, text_b=None, label='1')]       
        label_list = ["0", "1", "2", "3", "4"]
        eval_examples_len = len(eval_examples)        
        label_map = {label: i for i, label in enumerate(label_list)}
        path_to_model_directory = os.path.join(os.getcwd(),  "finetuned_BERT_model_files")
        tokenizer = BertTokenizer.from_pretrained(os.path.join(path_to_model_directory), do_lower_case=False)
        eval_examples_for_processing = [(example, label_map, max_seq_length, tokenizer, "classification") \
                                        for example in eval_examples]
        
        eval_features =  generate_eval_features(eval_examples_for_processing, eval_examples_len)                  
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        # since "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        print(all_label_ids)  
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)        
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size = eval_batch_size)     
        preds = []
        
        finetuned_model = linearClassifier.from_pretrained(path_to_model_directory,ignore_mismatched_sizes=True)  
        model = finetuned_model.eval()
        
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

             
            with torch.no_grad():

                output = model(input_ids, segment_ids, input_mask, labels=None)
                print(output)
                logits = output[0]
                
            if len(preds)==0:
                preds.append(logits.cpu().numpy())
            else: 
                preds[0] = np.append(preds[0], logits.cpu().numpy(), axis = 0)

        
        pred_list=preds[0].tolist()
             
        preds = preds[0]
        pred_probs = softmax(preds, axis=1)
        diff = abs(pred_probs[0][0]-pred_probs[0][1])
        predicted_class_probability = np.max(pred_probs)        
        predicted_class = np.argmax(pred_probs)
        if predicted_class==0:
            predicted_class=-1
        if diff<=0.2:
            predicted_class=0     
        #predicted_class = predicted_class - 2 
        return predicted_class, predicted_class_probability,diff

def load_model_and_tokenizer(path_to_model_directory):
        """ "path_to_directory" is the directory where the model elements
        (pytorch_model.bin, config.json and vocab.txt) are be stored """
        tokenizer = BertTokenizer.from_pretrained(os.path.join(path_to_model_directory), do_lower_case=False)
        finetuned_model = linearClassifier.from_pretrained(path_to_model_directory,ignore_mismatched_sizes=True)  
        model = finetuned_model.eval()
        
        return model, tokenizer
def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

if __name__ == '__main__':
    gu = GeneralUtils()
    current_datetime_timestamp = gu.getCurrentTimestsamp()
    from_datetime_timestamp = current_datetime_timestamp - gu.convertDaysToMiliseconds(decay_past_window)
    path_to_model_directory = os.path.join(os.getcwd(),  "finetuned_BERT_model_files")
    model, tokenizer=load_model_and_tokenizer(path_to_model_directory)
    #get_reports_sentiments( reports_df, model, tokenizer)
    init_path = os.path.join(os.getcwd(),"wetransfer_sifted_website_2023-05-09_1052","sifted_website")
    sent_list = []
    folds = [x[0] for x in os.walk(init_path)]
    path_to_model_directory = os.path.join(os.getcwd(),  "finetuned_BERT_model_files")
    tokenizer = BertTokenizer.from_pretrained(os.path.join(path_to_model_directory), do_lower_case=False)
    finetuned_model = linearClassifier.from_pretrained(path_to_model_directory,ignore_mismatched_sizes=True)  
    model = finetuned_model.eval()
    cnt=0
    min_diff = 1
    max_diff = 0
    for fold in folds:
        onlyfiles = [f for f in listdir(fold) if isfile(join(fold, f))]
        txt_files = filter(lambda x: x[-4:] == '.txt', onlyfiles)
        txt_files_1 = glob.glob("{}/*.txt".format(fold))
        #while cnt<=500:
        for txt_file in txt_files_1:
            cnt+=1
            try:
                curr_txt = open(txt_file,"r",encoding="utf8").readlines()
                curr_txt =" ".join([str(item) for item in curr_txt])
            except OSError as e:
                pass
                
            sent_json={}
            sent_json["Title"]=txt_file.split("\\")[-1].split('.')[0]
                
                
            predicted_txt_class, predicted_txt_class_probability,diff = predict_sentiment(curr_txt,path_to_model_directory)
            if diff<min_diff:
                min_diff=diff
            if diff>max_diff:
                max_diff=diff
            sent_json["Sentiment"]= predicted_txt_class
            sent_json["Prob"]= round(float(predicted_txt_class_probability),4)

            sent_list.append(sent_json)
            print(type(sent_list))
       
    
    print(min_diff)
    print(max_diff)
    print(((min_diff+max_diff)/2))
    with open('reports_sentiment.json', 'w') as f:
        
        json.dump(sent_list, f, default=np_encoder)
    f.close()