#!/usr/bin/env python
# coding: utf-8

# In[1]:


# explorations
'''
https://huggingface.co/osanseviero/full-sentence-distillroberta2 - can be used to select final set of sentences
'''


# In[2]:


import os
import sys
import argparse
import torch
import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
from openie import StanfordOpenIE
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import gensim.downloader
import gensim
from transformers import pipeline
import csv
import contractions
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.models import MultiTagger
import re
from operator import itemgetter
import math


# In[3]:


distilBertUnmasker = pipeline('fill-mask', model='distilbert-base-uncased')
nerTagger = pipeline("ner", grouped_entities=True)


# ## Read File

# In[4]:


with open("dialogues_train.txt","r") as f:
    daily_dialog_list = f.read().split("\n")


# In[5]:


conv_list = list()


# ### a. To append all parts of multi turn conversation

# In[117]:



for conv in daily_dialog_list:
    s_l = conv[:len(conv)-len(" __eou__ ")].split(" __eou__ ")
    i=0
    while( i < (len(s_l)-1)):
        tup = (s_l[i],s_l[i+1])
        i=i+2
        conv_list.append(tup)


# In[18]:


conv_list


# ### b. To append only 2 turns

# In[6]:



for conv in daily_dialog_list:
    s_l = conv[:len(conv)-len(" __eou__ ")].split(" __eou__ ")
    if(s_l[0] and s_l[1]):
        tup = (s_l[0],s_l[1])
    conv_list.append(tup)


# In[7]:


## Verify length
len(conv_list)


# ## Data Preprocessing and clean up

# In[8]:



conv_list =[(contractions.fix(tup[0]),contractions.fix(tup[1])) for tup in conv_list]
               
               
               


# ### a. Verify Data Structures

# In[9]:


conv_list[:100]


# ## Utility Functions

# ### Perplexity Scores

# In[10]:


import math
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)


# In[187]:


score('I lost wallet, I dont know')


# In[12]:


# utils

def getWordnetSynonymAndAntonynm(word):
    syn = list()
    ant = list()
    for synset in wordnet.synsets(word):
       for lemma in synset.lemmas():
          syn.append(lemma.name())    #add the synonyms
          if lemma.antonyms():    #When antonyms are available, add them into the list
              ant.append(lemma.antonyms()[0].name())
    return set(syn), set(ant)

def removeStopWords(string):
    filtered_words = [word for word in string.split() if word not in stopwords.words('english')]
    return filtered_words

posTagger = SequenceTagger.load("flair/pos-english")
def getPosTags(sentence):
    msentence = Sentence(sentence)

    # predict NER tags
    posTagger.predict(msentence)
    
    tags = [tag.labels[0].value for tag in msentence.get_spans('pos')]
    return list(zip(sentence.split(), tags))


# In[13]:


# https://huggingface.co/mrm8488/t5-base-finetuned-common_gen?text=tree+plant+ground+hole+dig
# https://github.com/INK-USC/CommonGen

from transformers import AutoModelWithLMHead, AutoTokenizer

commongen_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-common_gen")
commongen_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-common_gen")

def commongen_sentence(words, max_length=32):
  input_text = words
  features = commongen_tokenizer([input_text], return_tensors='pt')
  output = commongen_model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return commongen_tokenizer.decode(output[0], skip_special_tokens=True)

# test model to verify functionality
words = "tree plant ground hole dig"
commongen_sentence(words)

# load comet - atomic and conceptnet models

# atomic

atomic_model_file = 'pretrained_models/atomic_pretrained_model.pickle'
atomic_opt, atomic_state_dict = interactive.load_model_file(atomic_model_file)
atomic_data_loader, atomic_text_encoder = interactive.load_data("atomic", atomic_opt)
atomic_n_ctx = atomic_data_loader.max_event + atomic_data_loader.max_effect
atomic_n_vocab = len(atomic_text_encoder.encoder) + atomic_n_ctx
atomicModel = interactive.make_model(atomic_opt, atomic_n_vocab, atomic_n_ctx, atomic_state_dict)

cfg.device = "cpu"
sampling_algorithm = 'topk-4'
category = "all"
atomic_sampler = interactive.set_sampler(atomic_opt, sampling_algorithm, atomic_data_loader)
atomic_all_categories = atomic_data_loader.categories

# returns results sorted by beam_loss
def getAtomicResult(input_event, category):
    sequence_all = {}
    sequence_all["event"] = input_event
    sequence_all["effect_type"] = category

    with torch.no_grad():

        batch = interactive.set_atomic_inputs(
            input_event, category, atomic_data_loader, atomic_text_encoder)

        sampling_result = atomic_sampler.generate_sequence(
            batch, atomicModel, atomic_data_loader, atomic_data_loader.max_event +
            data.atomic_data.num_delimiter_tokens["category"],
            atomic_data_loader.max_effect -
            data.atomic_data.num_delimiter_tokens["category"])
    
    results = zip(sampling_result["beam_losses"], sampling_result["beams"])
    sortedResults = sorted(results, key=lambda x: x[0])
    sortedResults = [result[1] for result in sortedResults]
    return sortedResults

def getAtomicResults(input_event):
    results = {}
    for category in atomic_all_categories:
        res = getAtomicResult(input_event, category)
        results[category] = res
    return results

# concept net

conceptnet_model_file = 'pretrained_models/conceptnet_pretrained_model.pickle'
conceptnet_opt, conceptnet_state_dict = interactive.load_model_file(conceptnet_model_file)

conceptnet_data_loader, conceptnet_text_encoder = interactive.load_data("conceptnet", conceptnet_opt)

conceptnet_n_ctx = conceptnet_data_loader.max_e1 + conceptnet_data_loader.max_e2 + conceptnet_data_loader.max_r
conceptnet_n_vocab = len(conceptnet_text_encoder.encoder) + conceptnet_n_ctx

conceptnet_model = interactive.make_model(conceptnet_opt, conceptnet_n_vocab, conceptnet_n_ctx, conceptnet_state_dict)

conceptnet_sampler = interactive.set_sampler(conceptnet_opt, sampling_algorithm, conceptnet_data_loader)
conceptnet_relation = "all"
conceptnet_all_relations = data.conceptnet_data.conceptnet_relations

# returns results sorted by beam_loss
def getConceptNetResult(input_sentence, relation, force=False):
    sequence_all = {}

    sequence_all["e1"] = input_sentence
    sequence_all["relation"] = relation

    with torch.no_grad():
        if conceptnet_data_loader.max_r != 1:
            relation_sequence = data.conceptnet_data.split_into_words[relation]
        else:
            relation_sequence = "<{}>".format(relation)

        batch, abort = interactive.set_conceptnet_inputs(
            input_sentence, relation_sequence, conceptnet_text_encoder,
            conceptnet_data_loader.max_e1, conceptnet_data_loader.max_r, force)

        if abort:
            return {}

        sampling_result = conceptnet_sampler.generate_sequence(
            batch, conceptnet_model, conceptnet_data_loader,
            conceptnet_data_loader.max_e1 + conceptnet_data_loader.max_r,
            conceptnet_data_loader.max_e2)
    results = zip(sampling_result["beam_losses"], sampling_result["beams"])
    sortedResults = sorted(results, key=lambda x: x[0])
    sortedResults = [result[1] for result in sortedResults]
    return sortedResults

def getConceptNetResults(input_sentence):
    results = {}
    for relation in conceptnet_all_relations:
        results[relation] = getConceptNetResult(input_sentence, relation)
    return results


# In[14]:


posTagger = MultiTagger.load(['pos','ner'])


# ### Similarity Scores

# In[15]:



model_sim = SentenceTransformer('bert-base-nli-mean-tokens')


# In[16]:


sentences = [
    "Three years later, the coffin was still full of Jello.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "He found a leprechaun in his walnut shell."
]


# In[17]:


sentences = ['You know that is tempting but is really not good for our fitness .','i think i would go for a few beers after dinner.','if you want to take your fitness to the next level then you know where to go for a few beers after dinner.']


# In[18]:


sentence_embeddings = model_sim.encode(sentences)


# In[19]:


def get_sim_score(sentence_1, sentences):
    l = [sentence_1] + sentences
#     print(l)
    sentence_embeddings = model_sim.encode(l)
    cs_0 = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])
    l = list()
    i=0
    for v in cs_0[0]:
#         print(v)
        l.append([i,v])
        i=i+1
    return l
    
    


# In[20]:


x = "Three years later, the coffin was still full of Jello."
y = [ "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "He found a leprechaun in his walnut shell."]


# In[21]:


a = get_sim_score(x,y)


# In[22]:


a 


# ## Negative Sentence Generation

# ### a . Strategy 1

# In[3]:


#### For Noun - appends not before the word , and checks the 'Desires' relation from ConceptNet for replacement


# In[23]:


def remove_stopwords(sent_list):
    n_l =list()
    for s in sent_list:
        n_s = ' '.join([word for word in s.split() if word not in stopwords.words('english')])
        if(len(n_s)==0):
            continue
        n_l.append(n_s )
    return n_l


# In[24]:


def get_pos_tags(s):
    sentence = Sentence(s)
    posTagger.predict(sentence)
    d = sentence.to_dict(tag_type='pos')
    return d


# In[25]:


from collections import Counter


# ### Strategy 4 : Use Sim score too

# In[1]:


def transform_sent_multi_output(d,c):
    mod_flag = False
    new_sent = [""]
    index = 0
    while index < len(d['entities']):
            tag = d['entities'][index]
            label = str(tag['labels'][0]).split(" ")[0]
            word_tok = tag['text']
            if( not(mod_flag) and label == 'JJ'):
#                 c.update(['JJ'])
                mod_flag = True
                flag = False
                for synset in wordnet.synsets(tag['text']):
                    for lemma in synset.lemmas():
                        if lemma.antonyms():
                            ant = lemma.antonyms()[0].name()
                            flag = True
                            break
                    if(flag):
                        break
                if(flag):
                    word_negate = ant
                else:
                    word_negate = "not "+tag['text']
                word_tok = word_negate
                new_sent_c = list()
                for s in new_sent:
                    new_sent_c.append(s+" "+word_negate)
#                 new_sent = new_sent+ " "+word_tok
                new_sent = new_sent_c
                index =index + 1
            elif(not(mod_flag) and label == 'NN'):
                mod_flag = True
                word_negate = tag['text']
                infs = getConceptNetResult(word_negate,'Desires')
                infs_m = min(len(infs),3)
                infs = infs[0:infs_m]
                new_sent_c = list()
                for s in new_sent:
                    for inf in infs:
                        new_text = "not "+ inf
                        word_tok = new_text
                        new_sent_c.append(s+" "+word_tok)
#                 new_sent = new_sent+ " "+word_tok
                new_sent = new_sent_c
#                 new_sent = new_sent + " "+word_tok
                index =index + 1
            elif(not(mod_flag) and len(label) >=2 and label[0:2] == 'VB'):
                mod_flag = True
                min_ind = max(0,index-3)
                max_ind = min(len(d['entities'])-1,index+3)
                vb_context = ""
                min_cont = ""
                for j in range(min_ind,max_ind+1):
                    vb_context += " "+ d['entities'][j]['text']
                    if(j<index):
                        min_cont += d['entities'][j]['text']
                infs = getConceptNetResult(vb_context,'HasLastSubevent')
                infs_m = min(len(infs),3)
                infs = infs[0:infs_m]
                new_sent_c = list()
                for s in new_sent:
                    for inf in infs:
                        word_tok = inf +" then "+ vb_context
                        ns = s.replace(min_cont,"")
                        new_sent_c.append(ns+" "+word_tok)
                new_sent = new_sent_c      
#                 new_sent += " "+word_tok
                index =index + 3
            else:
                index +=1
                new_sent_c = list()
                for s in new_sent:
                    new_sent_c.append(s+" "+word_tok)
                new_sent = new_sent_c
#     print("new sent : ",new_sent)
    return (new_sent,mod_flag)


# In[2]:


mod_out = list()


# In[ ]:


for i in range(8000,9000):
    response = conv_list[i][1].strip()
    if(response[len(response)-1]=='.'):
         response = response[:-1]
    print("Ind : ",i," Time : ",datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    response_list = response.split('.')
    com_sense = ""
    new_res = ""
    com_flag = False
    prev_sent = remove_stopwords([conv_list[i][0]])[0]
    resp_list_wo_stopwords = response_list[:]
#     print("resp_list_wo_stopwords : ",resp_list_wo_stopwords)
    c = Counter()
    com_sent_ap_c=[""]
    for resp in resp_list_wo_stopwords:
        d = get_pos_tags(resp)
        (mod_sent_list,mod_flag) = transform_sent_multi_output(d,c)
        com_sent_ap = list()
        for cs in com_sent_ap_c:
            for ns in mod_sent_list:
                com_sent_ap.append(cs+" "+ns+ ".")
        com_sent_ap_c = com_sent_ap
    com_flag = com_flag or mod_flag
    if(com_flag):
        print("Actual Tuple : ",conv_list[i])   
        print("Malformed Sentences : ",com_sent_ap_c)
        print("Commonsense Output")
        com_resp = list()
        len_to_iter = min(len(com_sent_ap_c),10)
        strategy1 = False
        if(strategy1):
            for i1,s in enumerate(com_sent_ap_c):
                cr = commongen_sentence(s,50)
    #             print(cr)
    #             print(score(cr))
                com_resp.append([cr,score(cr)])
    #             print(i1," ",cr," ",score(cr))
        else:
            window_size =2
            for ind,sent in enumerate(com_sent_ap_c[:len_to_iter]):
                if(len(sent.strip()) == 0):
                    continue
                com_sent_ap_c_l = [prev_sent] + sent.split('.')[:-1]
                i2=1
                while(i2<(len(com_sent_ap_c_l))):
                    s=""
                    cr_l = ""
                    j=i2
                    while(j>=i2-window_size and j >= 0 ):
                        s =  com_sent_ap_c_l[j] + " "+s 
                        j=j-1
                    cr = commongen_sentence(s,50)
                    cr_l  = cr_l +cr+"."
                    i2+=1
                com_resp.append([cr_l,score(cr_l)])
                print("Output ",ind," ",cr_l," ",score(cr_l),get_sim_score(conv_list[i][0],[cr_l])[0][1])
        com_resp_l = [w[0] for w in com_resp]
        sim_score = get_sim_score(conv_list[i][0],com_resp_l)
        sim_score_sort = sorted(sim_score,key=itemgetter(1),reverse=True)
        com_resp_new = list()
        for w3 in range(min(len(sim_score_sort),10)):
            com_resp_new.append(com_resp[sim_score_sort[w3][0]]+[sim_score_sort[w3][0]]+[sim_score_sort[w3][1]])
        com_resp_perp_sort = sorted(com_resp_new,key=itemgetter(1))
        print(conv_list[i][0],"|",com_resp_perp_sort[0][0],com_resp_perp_sort[0][1],com_resp_perp_sort[0][2],com_resp_perp_sort[0][3])
        mod_out.append([conv_list[i][0],com_resp_perp_sort[0][0],com_resp_perp_sort[0][1],com_sent_ap_c[com_resp_perp_sort[0][2]]])
        file_save_iter =10
        if(count_print%file_save_iter ==0 ):
            with open("output_5.txt","a") as f , open("output_malformed__5.txt","a") as f2:
                for e_i in range(watermark,len(mod_out)):
                    element = mod_out[e_i]
                    try:
                        f.write(element[0]+" __eou__ "+element[1]+"\n")
                        f2.write(element[0]+"\t | \t"+element[1]+ "\t | \t"+element[3]+'\t | \t'+str(element[2])+"\n")
                    except:
                        print("Exception occured. Moving on to next one.")
                watermark = len(mod_out)
            f.close()
            f2.close()
        count_print +=1
    
    


# In[174]:


f1.close()
f2.close()

