pip install transformers==3

import torch
from torch import nn, optim
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import json
from sklearn.metrics import confusion_matrix

from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels = 2,
                                                      output_attentions = False,
                                                      output_hidden_states = False)
model.load_state_dict(torch.load("common_sense_classifier_final.model"))
model.cuda()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

usr_dataset = json.load(open("tc_usr_data.json"))
pc_dataset = json.load(open("pc_usr_data.json"))

usr_tuples = []

for el in usr_dataset:
    sent1 = el['context']
    sent1_split = []
    for line in sent1.splitlines(keepends=False):
        if len(line) != 0:
            sent1_split.append(line)
    
    for resp in el['responses']:
        sent2 = resp['response']
        conv = [sent1_split[len(sent1_split) - 1].rstrip(), sent2.rstrip()]
        usr_tuples.append(conv)

for el in pc_dataset:
    sent1 = el['context']
    sent1_split = []
    for line in sent1.splitlines(keepends=False):
        if len(line) != 0:
            sent1_split.append(line)
    
    for resp in el['responses']:
        sent2 = resp['response']
        conv = [sent1_split[len(sent1_split) - 1].rstrip(), sent2.rstrip()]
        usr_tuples.append(conv)

class CustomUSRDataset(Dataset):  
    def __init__(self, dataset, tokenizer, max_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        train_tuple = self.dataset[idx]
        if len(train_tuple) == 2:
          sentence = train_tuple[0] + " " + train_tuple[1]
        else:
          sentence = train_tuple
        encoding = self.tokenizer.encode_plus(sentence,
                                              add_special_tokens=True,
                                              max_length=self.max_len,
                                              truncation=True,
                                              return_token_type_ids=False,
                                              pad_to_max_length=True,
                                              return_attention_mask=True,
                                              return_tensors='pt')
        return {'index': idx,
                'conv_text': sentence,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

usr_data = CustomUSRDataset(usr_tuples, tokenizer, 150)
usr_loader = DataLoader(usr_data, batch_size = 1)

model.eval()
conf_val_list = []
for data in usr_loader:
    outputs = model(input_ids = torch.as_tensor(data['input_ids'], device = device),
                    attention_mask = torch.as_tensor(data['attention_mask'], device = device))
    preds = outputs[0]
    conf_val = preds.detach().cpu().numpy()
    conf_val_list.append(conf_val)

softmax = nn.Softmax(dim = 1)

pred_conf_val = []
for conf_val in conf_val_list:
    softmax_conf = softmax(torch.as_tensor(conf_val))
    softmax_conf = softmax_conf.detach().cpu().numpy()
    pred = np.argmax(softmax_conf, axis= 1)
    pred_conf_val.append([softmax_conf[0][1], pred[0]])

def GetBucket(num):
    num = num * 10
    if 0 <= num and num < 3.4:
        return 0
    elif 3.4 <= num and num < 6.7:
        return 1
    else:
        return 2

engage_list = []
context_list = []
natural_list = []
overall_list = []
understandable_list = []
knowledge_list = []

for el in usr_dataset:
    for resp in el['responses']:
        engage_scores = resp['Engaging']
        context_scores = resp['Maintains Context']
        natural_scores = resp['Natural']
        overall_scores = resp['Overall']
        understanblee_scores = resp['Understandable']
        knowledge_scores = resp['Uses Knowledge']

        score = (sum(engage_scores) / len(engage_scores)) / 3
        engage_list.append(GetBucket(score))

        score = (sum(context_scores) / len(context_scores)) / 3
        context_list.append(GetBucket(score))

        score = (sum(natural_scores) / len(natural_scores)) / 3
        natural_list.append(GetBucket(score))

        score = (sum(overall_scores) / len(overall_scores)) / 5
        overall_list.append(GetBucket(score))

        score = (sum(understanblee_scores) / len(understanblee_scores)) / 1
        understandable_list.append(GetBucket(score))

        score = (sum(knowledge_scores) / len(knowledge_scores)) / 1
        knowledge_list.append(GetBucket(score))

for el in pc_dataset:
    for resp in el['responses']:
        engage_scores = resp['Engaging']
        context_scores = resp['Maintains Context']
        natural_scores = resp['Natural']
        overall_scores = resp['Overall']
        understanblee_scores = resp['Understandable']
        knowledge_scores = resp['Uses Knowledge']

        score = (sum(engage_scores) / len(engage_scores)) / 3
        engage_list.append(GetBucket(score))

        score = (sum(context_scores) / len(context_scores)) / 3
        context_list.append(GetBucket(score))

        score = (sum(natural_scores) / len(natural_scores)) / 3
        natural_list.append(GetBucket(score))

        score = (sum(overall_scores) / len(overall_scores)) / 5
        overall_list.append(GetBucket(score))

        score = (sum(understanblee_scores) / len(understanblee_scores)) / 1
        understandable_list.append(GetBucket(score))

        score = (sum(knowledge_scores) / len(knowledge_scores)) / 1
        knowledge_list.append(GetBucket(score))

usr_conf_val_list = []
usr_pred_list = []
for pred_conf in pred_conf_val:
    usr_conf_val_list.append(GetBucket(pred_conf[0]))
    usr_pred_list.append(pred_conf[1])

engage_mat = confusion_matrix(engage_list, usr_conf_val_list, labels=[0,1,2])
df_cm = pd.DataFrame(engage_mat)
plt.figure(figsize=(12,8))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 30}, fmt=".3g") # font size
plt.ylabel("Engaging Score")
plt.xlabel("Sensibility Scores")
plt.xlim(0, 3)
plt.ylim(0 ,3)
plt.show()

mat = confusion_matrix(context_list, usr_conf_val_list, labels=[0,1,2])
df_cm = pd.DataFrame(mat)
plt.figure(figsize=(12,8))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 30}, fmt=".3g") # font size
plt.ylabel("Context Score")
plt.xlabel("Sensibility Scores")
plt.xlim(0, 3)
plt.ylim(0 ,3)
plt.show()

mat = confusion_matrix(natural_list, usr_conf_val_list, labels=[0,1,2])
df_cm = pd.DataFrame(mat)
plt.figure(figsize=(12,8))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 30}, fmt=".3g") # font size
plt.ylabel("Natural Score")
plt.xlabel("Sensibility Scores")
plt.xlim(0, 3)
plt.ylim(0 ,3)
plt.show()