import math
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from operator import itemgetter
import torch

# read generated candidates and select candidates for creating dataset
# choose candidate file name and final dataset file name to be created below

#datasetCandidateFile = 'generated_sentences_commongen.txt'
datasetCandidateFile = 'generated_sentences_discol.txt'
datasetFile = 'generated_dataset.txt'


pp_model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
pp_model.eval()
pp_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

def pp_score(sentence):
    tokenize_input = pp_tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([pp_tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = pp_model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)



model_sim = SentenceTransformer('bert-base-nli-mean-tokens')

def get_sim_score(sentence_1, sentences):
    l = [sentence_1] + sentences
    sentence_embeddings = model_sim.encode(l)
    cs_0 = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])
    l = list()
    i=0
    for v in cs_0[0]:
        l.append([i,v])
        i=i+1
    return l


#f = open(datasetCandidateFile, 'r', encoding='utf-8')
f = open(datasetCandidateFile, 'r', encoding='cp1252')
lines = f.read().split('\n')
f.close()
sentence_candidates = []
tempcollection = []
for x in lines:    
    if x == '':
        sentence_candidates.append(tempcollection)
        tempcollection = []
    else:
        tempcollection.append(x)
if tempcollection:
    sentence_candidates.append(tempcollection)


c = 0
for i in sentence_candidates:
    sentence_A, sentence_B = i[0].split(' __eou__ ')
    candidates = i[1:]

    sim_scores = get_sim_score(sentence_A, candidates)
    sim_scores_sort = sorted(sim_scores,key=itemgetter(1),reverse=True)
    if(len(sim_scores_sort)>1):
        sim_score_sort_n = sim_scores_sort[1:]
    else:
        sim_score_sort_n = sim_scores_sort[:]
    com_resp_new = list()
    for j in range(min(len(sim_score_sort_n),4)):
        com_resp_new.append([candidates[sim_score_sort_n[j][0]]]+[pp_score(candidates[sim_score_sort_n[j][0]])]+[sim_score_sort_n[j][1]])
    com_resp_perp_sort = sorted(com_resp_new,key=itemgetter(1))

    with open(datasetFile, "a", encoding='utf-8') as f1:
        if len(com_resp_perp_sort) < 2:
            f1.write(sentence_A+" __eou__ "+com_resp_perp_sort[0][0]+"\n")
        else:
            f1.write(sentence_A+" __eou__ "+com_resp_perp_sort[0][0]+"\n"+sentence_A+" __eou__ "+com_resp_perp_sort[1][0]+"\n")
    f1.close()
    print(c)
    c=c+1