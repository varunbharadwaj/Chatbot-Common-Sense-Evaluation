import os
import sys
import argparse
import torch
import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import gensim.downloader
import gensim
from transformers import pipeline
import requests
import json
import contractions
import time
from transformers import AutoModelWithLMHead, AutoTokenizer
from data_utils import *


# load comet - atomic and conceptnet models
# the following code for comet is taken from - https://github.com/atcbosselut/comet-commonsense and modified slightly to suit our needs
# modify the paths accordingly after following the setup instructions mentioned in the repository


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


invertAtomicQueryIntents = ['oReact', 'xIntent']   

def getInvertedAtomicIntentsFromQuery(phrase):
    results = {'context':[]}
    for intent in invertAtomicQueryIntents:
        atomicIntent = getAtomicResult(phrase, intent)
        atomicIntent = list(filter(lambda x: x!='none', atomicIntent))
        
        # add a 'not' prefix for reaction and intent to negate it if multi word ; else use antonym
        if intent in ['oReact', 'xIntent']:
            results['orig_'+intent] = atomicIntent
            negatedAtomicIntent = [negatedIntent for aIntent in atomicIntent for negatedIntent in getAntonymOrNegation(aIntent)]
            atomicIntent = [intent for aIntent in atomicIntent]
            results[intent] = negatedAtomicIntent
    return results


invertConceptnetQueryIntents = ['NotDesires', 'HasProperty', 'HasFirstSubevent', 'HasLastSubevent']
conceptNetContext = ['InheritsFrom']

# adding a 'not' to the intents to negate it
def getInvertedConceptNetIntentsFromQuery(phrase):
    results = {}
    for intent in invertConceptnetQueryIntents:
        conceptnetIntent = getConceptNetResult(phrase, intent)
        conceptnetIntent = list(filter(lambda x: x!='none', conceptnetIntent))
        
        # add a 'not' for reaction and intent to negate it
        if intent in ['HasProperty']:
            negatedIntent = [negatedIntent for aIntent in conceptnetIntent for negatedIntent in getAntonymOrNegation(aIntent)]
            results[intent] = negatedIntent
        if intent in ['NotDesires']:
            negatedIntent = [negatedIntent for aIntent in conceptnetIntent for negatedIntent in getAntonymOrNegation(aIntent)]
            results['invertedNotDesires'] = negatedIntent
        if intent in ['NotDesires', 'HasFirstSubevent', 'HasLastSubevent']:
            negatedIntent = [aIntent for aIntent in conceptnetIntent]
            results[intent] = negatedIntent
    
    for intent in conceptNetContext:
        conceptnetIntent = getConceptNetResult(phrase, intent)
        conceptnetIntent = list(filter(lambda x: x!='none', conceptnetIntent))
        results['context'] = conceptnetIntent
        
    return results

def getInvertedIntents(sentence):
    results = {}
    atomicResults = getInvertedAtomicIntentsFromQuery(sentence)
    conceptnetResults = getInvertedConceptNetIntentsFromQuery(sentence)
    for key in atomicResults:
        results[key] = atomicResults[key]
    for key in conceptnetResults:
        results[key] = conceptnetResults[key]
        
    return results































