pip install transformers==3

import torch
from torch import nn, optim
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
import time

pos_conv_list = []
neg_conv_list = []
dialogue_sep_token = " __eou__ "
topical_sep_token = "__eou__"

data_path = 'dialogues.txt'

with open(data_path,"r") as f:
    daily_dialog_list = f.readlines()
    lines = [line.rstrip() for line in daily_dialog_list]
    count = 0
    for line in lines:
      arr_line = line[:len(line)-len(dialogue_sep_token)].split(dialogue_sep_token)
      if len(arr_line) < 2:
          continue
      tup = (arr_line[0], arr_line[1])
      pos_conv_list.append(tup)

data_path = 'topical_chat_pairs.txt'

with open(data_path,"r") as f:
    topical_chat_list = f.readlines()
    lines = [line.rstrip() for line in topical_chat_list]
    count = 0
    for line in lines:
      arr_line = line[:len(line)-len(topical_sep_token)].split(topical_sep_token)
      if len(arr_line) < 2:
          continue
      tup = (arr_line[0], arr_line[1])
      pos_conv_list.append(tup)

data_path = 'non_common_sense_dataset.txt'

with open(data_path,"r") as f:
    neg_sense_list = f.readlines()
    lines = [line.rstrip() for line in neg_sense_list]
    count = 0
    for line in lines:
      arr_line = line.split(dialogue_sep_token)
      if len(arr_line) < 2:
          continue
      tup = (arr_line[0], arr_line[1])
      neg_conv_list.append(tup)

sense_dataset = []
for conv in pos_conv_list:
    sense_dataset.append([conv, 1])

for conv in neg_conv_list:
    sense_dataset.append([conv, 0])

random.shuffle(sense_dataset)

train_test_split_index = (len(sense_dataset)*80/100)
train_test_split_index = int(train_test_split_index)
train_test_split_index

train_dataset = sense_dataset[:train_test_split_index]
test_dataset = sense_dataset[train_test_split_index:]

class CustomTrainDataset(Dataset):  
    def __init__(self, dataset, tokenizer, max_len):
        self.dataset = dataset
        
        self.class_weights = [0,0]
        for sample in self.dataset:
            if sample[1] == 0:
                self.class_weights[0] += 1
            elif sample[1] == 1:
                self.class_weights[1] += 1
        self.class_weights[0]/=len(self.dataset)
        self.class_weights[1]/=len(self.dataset)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        train_tuple = self.dataset[idx][0]
        train_target = self.dataset[idx][1]
        sentence = train_tuple[0] + " " + train_tuple[1]
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
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(train_target, dtype=torch.long)}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_data = CustomTrainDataset(train_dataset, tokenizer, 125)

test_data = CustomTrainDataset(test_dataset, tokenizer, 125)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels = 2,
                                                      output_attentions = False,
                                                      output_hidden_states = False)
model.cuda()

train_loader = DataLoader(train_data, batch_size = 16, shuffle= True)
test_loader = DataLoader(test_data, batch_size = 1)

# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

optimizer = optim.AdamW(model.parameters(),lr=1e-05)

epochs = 2

total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(train_data.class_weights, device = device))

model.train()

losses = []

for epoch in range(epochs):
    start = time.time()

    print("Epoch: ",epoch + 1)
    for data in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids = torch.as_tensor(data['input_ids'], device = device),
                        attention_mask = torch.as_tensor(data['attention_mask'], device = device))
        loss = loss_fn(outputs[0], torch.as_tensor(data['targets'], device = device))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
    print("Loss:",sum(losses)/len(losses))
    end = time.time()
    print(end-start)

torch.save(model.state_dict(), 'common-sense_classifier.model')

conf_val_list = []

model.eval()
prediction = []
truth = []
for data in test_loader:
    outputs = model(input_ids = torch.as_tensor(data['input_ids'], device = device),
                    attention_mask = torch.as_tensor(data['attention_mask'], device = device))
    preds = outputs[0]
    conf_val = preds.detach().cpu().numpy()
    conf_val_list.append(conf_val)
    preds = torch.argmax(preds, dim = 1)
    preds = preds.detach().cpu().numpy()
    targets = data['targets'].detach().cpu().numpy()
    for i in range(len(preds)):
        prediction.append(preds[i])
        truth.append(targets[i])

from sklearn.metrics import classification_report
target_names = ['Non-Common-Sense', 'Common-Sense']

print(classification_report(truth, prediction, target_names=target_names))
