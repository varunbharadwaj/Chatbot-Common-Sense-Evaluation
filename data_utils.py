import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from flair.data import Sentence
from flair.models import SequenceTagger
import requests
import json
import contractions
import time
import sys
import os
from transformers import pipeline
from transformers import AutoModelWithLMHead, AutoTokenizer


# load and prepare models
sys.path.append(os.getcwd())
nerTagger = pipeline("ner", grouped_entities=True)


# POS tagger - flair
# https://huggingface.co/flair/pos-english
# load tagger
posTagger = SequenceTagger.load("flair/pos-english")



# google t5 transformer finetuned on CommonGen - generate sentences given set of concepts
# https://huggingface.co/mrm8488/t5-base-finetuned-common_gen?text=tree+plant+ground+hole+dig
# https://github.com/INK-USC/CommonGen

commongen_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-common_gen")
commongen_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-common_gen")

def commongen_sentence(words, max_length=32):
  input_text = words
  features = commongen_tokenizer([input_text], return_tensors='pt')
  output = commongen_model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return commongen_tokenizer.decode(output[0], skip_special_tokens=True)


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


def getPosTags(sentence):
    msentence = Sentence(sentence)

    # predict NER tags
    posTagger.predict(msentence)
    
    tags = [tag.labels[0].value for tag in msentence.get_spans('pos')]
    return list(zip(sentence.split(), tags))


def expandContactions(sentence):
    sentence = sentence.replace(" ’ ", "'")
    expanded_words = []    
    for word in sentence.split():
      # using contractions.fix to expand the shotened words
      expanded_words.append(contractions.fix(word))  
    expanded_text = ' '.join(expanded_words)
    return expanded_text


def preprocessPhrases(phrase):
    phrase = phrase.replace('.', '')
    phrase = phrase.strip()
    return phrase


def getAntonymOrNegation(phrase):
    phrase = preprocessPhrases(phrase)
    if len(phrase.split()) > 1:
        # negate phrase by adding 'not' prefix
        return ['not {}'.format(phrase)]
    else:
        # find antonym if possible
        syn, ant = getWordnetSynonymAndAntonynm(phrase)
        if ant:
            ant = [x.replace('_', ' ') for x in ant]
            return ant
        else:
            # if antonym not present, ignore the word
            return []
        
retainPosTags = set(['JJ', 'FW', 'JJR', 'JJS', 'NN', 'NNP', 'NNPS', 'NNS', 'RB', 'RBR', 'RBS', 'RP', 'VB', 'VBG', 'VBN', 'VBP', 'VBZ', 'VBD', 'WRB', 'XX'])
lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize(word):
    return lemmatizer.lemmatize(word)

# returns the important words in the sentence
def getCentralWords(sentence):
    nerTags = nerTagger(sentence)
    entities = set([x['word'] for x in nerTags])
    posTags = getPosTags(sentence)
    words = set(removeStopWords(sentence))
    resWords = []
    for w in posTags:
        if w[1] in retainPosTags or w[0] in entities:
            resWords.append(lemmatize(w[0]))
    
    # create 6-grams to avoid very long sentences
    if len(resWords) < 6:
        return [' '.join(resWords)]
    sixgrams = []
    for i in range(len(resWords)-5):
        sixgrams.append(' '.join((resWords[i],resWords[i+1],resWords[i+2], resWords[i+3], resWords[i+4], resWords[i+5])))
    
    return sixgrams


# read dialogues from dialy dialog and topical chat datasets 
# we pick the first set of query-response from every conversation (first two turns)
def getConversationData():
	conv_list_daily_dialog = list()
	dailyDialogFile = "dataset/daily_dialogue.txt"

	with open(dailyDialogFile, "r", encoding="utf8") as f:
	    daily_dialog_list = f.read().split("\n")
	    for conv in daily_dialog_list:
	        convPairs = conv[:len(conv)-len(" __eou__ ")].split(" __eou__ ")
	        i=0
	        if i < (len(convPairs)-1):
	            tup = (convPairs[i],convPairs[i+1])
	            conv_list_daily_dialog.append(tup)

	conv_list_topical_chat = list()
	topicalChatFile = "dataset/topical_chat_pairs.txt"
	with open(topicalChatFile, "r", encoding="utf8") as f:
	    topical_chat_list = f.read().split("\n")
	    for conv in topical_chat_list:
	        convPairs = conv[:len(conv)-len("__eou__")].split("__eou__")
	        if len(convPairs) > 1:
	            tup = (convPairs[0],convPairs[1])
	            conv_list_topical_chat.append(tup)
	            
	conv_list = conv_list_daily_dialog
	conv_list.extend(conv_list_topical_chat)
	return conv_list
