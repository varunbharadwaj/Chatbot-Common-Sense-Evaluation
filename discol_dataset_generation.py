from data_utils import *
from comet_utils import *


# A and B here refer to query and response in the conversation (turn 1 and turn 2)

# update the discol url after starting discol server
discolURL = '<discol_url>'

numTopIntents = 1
numTopContext = 2
generationLengthSlack = 5
numVerbsAndNN = 2
conv_list = getConversationData()

# make a call to discol server to get response given history and set of keywords
# error token is returned in case of any exception
def getDiscolResponse(history, keywords):
    try:
        headers = {
        'content-type' : 'application/json',
        }
        payload = {"history": history, "keywords": keywords, "temperature": "0.7","top-k": "5","top-p": "0.7"}
        response = requests.post(discolURL, data=json.dumps(payload), allow_redirects=True, headers=headers)
        resp = response.json()['responses'][0]
        return resp
    except Exception as e:
        print(e)
        return '<ERROR>'


def generateDiscolSentenceFromConcepts(concepts, history=[]):
    response = getDiscolResponse(history, concepts)
    return response


def generateDiscolSentencesFromVBAndNNIntents(intentA, intentB, sentenceA, sentenceB, tagsA, tagsB, maxLen=32):
    nounsInB = ' '.join([x[0] for x in filter(lambda x: 'NN' in x[1], tagsB)])
    nounInA = [x[0] for x in filter(lambda x: 'NN' in x[1], tagsA)]
    if nounInA:
        nounInA = nounInA[0]
    else:
        nounInA = ''
    
    #conversation_history = conv_context[sentenceA]
    sentencelenB = len(sentenceB)
    
    # experiment swapping intents - gave better results
    intentA, intentB = intentB, intentA
    
    # intents from A
    reactionA = []
    if 'oReact' in intentA:
        reactionA = [x for x in intentA['oReact']][:numTopIntents]
        
    lastEventA = []
    if 'HasLastSubevent' in intentA:
        lastEventA = [x for x in intentA['HasLastSubevent']][:numTopIntents]
        
    notDesiresA = []
    if 'NotDesires' in intentA:
        notDesiresA = [x for x in intentA['NotDesires']][:numTopIntents]
        
    hasPropertyA = []
    if 'HasProperty' in intentA:
        hasPropertyA = [x for x in intentA['HasProperty']][:numTopIntents]
        
    origIntentA = []
    if 'orig_xIntent' in intentA:
        origIntentA = [x for x in intentA['orig_xIntent']][:numTopIntents]
        
    firstEventA = []
    if 'HasFirstSubevent' in intentA:
        firstEventA = [x for x in intentA['HasFirstSubevent']][:numTopIntents]
        
    xIntentA = []
    if 'xIntent' in intentA:
        xIntentA = [x for x in intentA['xIntent']][:numTopIntents]
        
    contextA = []
    if 'context' in intentA:
        contextA = [x for x in intentA['context']][:numTopContext]
    
    # intents from B
    notDesiresB = []
    if 'NotDesires' in intentB:
        notDesiresB = [x for x in intentB['NotDesires']][:numTopIntents]
        
    xIntentB = []
    if 'xIntent' in intentB:
        xIntentB = [x for x in intentB['xIntent']][:numTopIntents]
        
    firstEventB = []
    if 'HasFirstSubevent' in intentB:
        firstEventB = [x for x in intentB['HasFirstSubevent']][:numTopIntents]
        
    lastEventB = []
    if 'HasLastSubevent' in intentB:
        lastEventB = [x for x in intentB['HasLastSubevent']][:numTopIntents]
        
    reactionB = []
    if 'oReact' in intentB:
        reactionB = [x for x in intentB['oReact']][:numTopIntents]
        
    hasPropertyB = []
    if 'HasProperty' in intentB:
        hasPropertyB = [x for x in intentB['HasProperty']][:numTopIntents]
        
    contextB = []
    if 'context' in intentB:
        contextB = [x for x in intentB['context']][:numTopContext]
        
    candidates = []
    
    for desiresA in notDesiresA:
        for propertyA in hasPropertyA:
            for lastEvent in lastEventA:
                combinedContextAndIntent = '{} {} {} {} {}'.format(nounsInB, nounInA, desiresA, propertyA, lastEvent)
                keywords = [nounsInB, nounInA, desiresA, propertyA, lastEvent]
                keywords = sentenceB.split() + keywords
                generated_sentence = generateDiscolSentenceFromConcepts(keywords, [sentenceA, sentenceB])
                candidates.append([combinedContextAndIntent, generated_sentence])
                time.sleep(1)

    for ointentA in origIntentA:
        for lasteventA in lastEventA:
                for lasteventB in lastEventB:
                    combinedContextAndIntent = '{} {} {} {} {}'.format(nounsInB, nounInA, ointentA, lasteventA, lasteventB)
                    keywords = [nounsInB, nounInA, ointentA, lasteventA, lasteventB]
                    keywords = sentenceB.split() + keywords
                    generated_sentence = generateDiscolSentenceFromConcepts(keywords, [sentenceA, sentenceB])
                    candidates.append([combinedContextAndIntent, generated_sentence])
                    time.sleep(1)
            
    for ointentA in origIntentA:
        for context in contextA:
            combinedContextAndIntent = '{} {} {} {}'.format(nounsInB, nounInA, ointentA, context)
            keywords = [nounsInB, nounInA, ointentA, context]
            keywords = sentenceB.split() + keywords
            generated_sentence = generateDiscolSentenceFromConcepts(keywords, [sentenceA, sentenceB])
            candidates.append([combinedContextAndIntent, generated_sentence])
            time.sleep(1)
        
    for desiresA in notDesiresA:
        for ointentA in origIntentA:
            combinedContextAndIntent = '{} {} {} {}'.format(nounsInB, nounInA, ointentA, desiresA)
            keywords = [nounsInB, nounInA, ointentA, desiresA]
            keywords = sentenceB.split() + keywords
            generated_sentence = generateDiscolSentenceFromConcepts(keywords, [sentenceA, sentenceB])
            candidates.append([combinedContextAndIntent, generated_sentence])
            time.sleep(1)
                    
    for ointentA in origIntentA:
        for lasteventB in lastEventB:
            combinedContextAndIntent = '{} {} {} {}'.format(nounsInB, nounInA, ointentA, lasteventB)
            keywords = [nounsInB, nounInA, ointentA, lasteventB]
            keywords = sentenceB.split() + keywords
            generated_sentence = generateDiscolSentenceFromConcepts(keywords, [sentenceA, sentenceB])
            candidates.append([combinedContextAndIntent, generated_sentence])
            time.sleep(1)
            
    for intentB in xIntentB:
            for context in contextB:
                for desireB in notDesiresB:
                    combinedContextAndIntent = '{} {} {} {}'.format(nounsInB, context, intentB, desireB)
                    keywords = [nounsInB, context, intentB, desireB]
                    keywords = sentenceB.split() + keywords
                    generated_sentence = generateDiscolSentenceFromConcepts(keywords, [sentenceA, sentenceB])
                    candidates.append([combinedContextAndIntent, generated_sentence])
                    time.sleep(1)

    return candidates


def getInvertedSentenceCandidatesFromDiscol(sentenceA, sentenceB):
    candidates = []
    tagsA = getPosTags(sentenceA)
    tagsB = getPosTags(sentenceB)
    wordsA = getCentralWords(sentenceA)
    wordsB = getCentralWords(sentenceB)
    verbAndNNInA = [x[0] for x in filter(lambda x: x[0] in wordsA and 'VB' in x[1] or 'NN' in x[1], tagsA)][:numVerbsAndNN]
    verbAndNNInA = ' '.join(verbAndNNInA)
    verbAndNNInB = [x[0] for x in filter(lambda x: x[0] in wordsB and'VB' in x[1] or 'NN' in x[1], tagsB)][:numVerbsAndNN]
    verbAndNNInB = ' '.join(verbAndNNInB)
    
    if verbAndNNInA and verbAndNNInB:
        intentA = getInvertedIntents(verbAndNNInA)
        intentB = getInvertedIntents(verbAndNNInB)
        sentences = generateDiscolSentencesFromVBAndNNIntents(intentA, intentB, sentenceA, sentenceB, tagsA, tagsB, len(sentenceB))
        # get generated candidates
        sentences = [x[1] for x in sentences]
        candidates.extend(sentences)
    else:
        print('No NN/VB found. Skipping sentence.')
#		  removing fallback to commongen
#         wordsA = getCentralWords(sentenceA)
#         wordsB = getCentralWords(sentenceB)
#         context = []
#         for words in wordsB:
#             processedSentenceContext = words
#             intents = getInvertedIntents(words)
#             sentences = generateSentencesFromIntents(intents, context, sentenceB, len(sentenceA))
#             candidates.extend(sentences)

    return candidates



# candidate generation using discol
# candidate file format - SentenceA (query) __eou__ SentenceB (response) followed by candidates in next line
# empty line separator between conversations

def generateDiscolCandidatesAndWriteToFile(conv_list, output_file='generated_sentences_discol.txt', error_file='discol_error_file.txt'):
    generatedResults = []
    errors = []
    i = 1
    for sentenceA, sentenceB in conv_list:
        expandedSentenceA = expandContactions(sentenceA)
        expandedSentenceB = expandContactions(sentenceB)
        print(i)
        i += 1
        try:
            results = getInvertedSentenceCandidatesFromDiscol(expandedSentenceA, expandedSentenceB)
            if results:
                generatedResults.append('{}\n{}'.format('{} __eou__ {}'.format(sentenceA, sentenceB), '\n'.join(results)))
        except Exception as e:
            errors.append('{}\n{}'.format(sentenceA, sentenceB))
            print(e)
        time.sleep(1)

    with open(output_file, 'w')as f:
        f.write('\n\n'.join(generatedResults))
        
    with open(error_file, 'w')as f:
        f.write('\n\n'.join(errors))


if __name__ == '__main__':     
	generateDiscolCandidatesAndWriteToFile(conv_list)




















