from data_utils import *
from comet_utils import *


conv_list = getConversationData()


# A and B here refer to query and response in the conversation (turn 1 and turn 2)

numTopIntents = 1
numTopContext = 2
generationLengthSlack = 5
numVerbsAndNN = 2

def generateSentencesFromIntents(intents, context, sentenceContext, maxLen=32):
    contexts = context[:numTopContext]
    reactions = []
    if 'oReact' in intents:
        reactions = [x for x in intents['oReact']][:numTopIntents]
        
    xintent = []
    if 'xIntent' in intents:
        xintent = [x for x in intents['xIntent']][:numTopIntents]
        
    desires = []
    if 'invertedNotDesires' in intents:
        desires = [x for x in intents['invertedNotDesires']][:numTopIntents]
        
    hasProperty = []
    if 'HasProperty' in intents:
        hasProperty = [x for x in intents['HasProperty']][:numTopIntents]
        
    candidates = []
    for intent in list(zip(xintent, desires)):
        intentStr, desire = intent
        combinedContextAndIntent = '{} {} {}'.format(intentStr, sentenceContext, desire)
        generated_sentence = commongen_sentence(combinedContextAndIntent, min(32, maxLen+generationLengthSlack))
        candidates.append(generated_sentence)
    
    return candidates


def generateSentencesFromVBAndNNIntents(intentA, intentB, sentenceA, sentenceB, tagsA, tagsB, maxLen=32):
    nounsInB = ' '.join([x[0] for x in filter(lambda x: 'NN' in x[1], tagsB)])
    nounInA = [x[0] for x in filter(lambda x: 'NN' in x[1], tagsA)]
    if nounInA:
        nounInA = nounInA[0]
    else:
        nounInA = ''
    
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
        contextB = [x for x in intentB['context']][:numTopIntents]
    
    candidates = []

    # explore combination of intents from above
    for desiresA in notDesiresA:
        for propertyA in hasPropertyA:
            for lastEvent in lastEventA:
                combinedContextAndIntent = '{} {} {} {} {} {}'.format(nounsInB, nounInA, desiresA, sentenceA, propertyA, lastEvent)
                generated_sentence = commongen_sentence(combinedContextAndIntent, min(32, maxLen+generationLengthSlack))
                candidates.append(generated_sentence)

    for ointentA in origIntentA:
        for lasteventA in lastEventA:
            for firsteventB in firstEventB:
                for lasteventB in lastEventB:
                    combinedContextAndIntent = '{} {} {} {} {} {} {}'.format(nounsInB, nounInA, ointentA, sentenceA, lasteventA, firsteventB, lasteventB)
                    generated_sentence = commongen_sentence(combinedContextAndIntent, min(32, maxLen+generationLengthSlack))
                    candidates.append(generated_sentence)
    
    for reaction in reactionA:
        for lastEvent in lastEventA:
            combinedContextAndIntent = '{} {} {} {} {}'.format(nounsInB, nounInA, reaction, sentenceB, lastEvent)
            generated_sentence = commongen_sentence(combinedContextAndIntent, min(32, maxLen+generationLengthSlack))
            candidates.append(generated_sentence)
            
    for desiresA in notDesiresA:
        for propertyA in hasPropertyA:
            for lastEvent in lastEventA:
                combinedContextAndIntent = '{} {} {} {} {} {}'.format(nounsInB, nounInA, desiresA, sentenceB, propertyA, lastEvent)
                generated_sentence = commongen_sentence(combinedContextAndIntent, min(32, maxLen+generationLengthSlack))
                candidates.append(generated_sentence)
                
    for ointentA in origIntentA:
        for lasteventA in lastEventA:
            for firsteventB in firstEventB:
                for lasteventB in lastEventB:
                    combinedContextAndIntent = '{} {} {} {} {} {} {}'.format(nounsInB, nounInA, ointentA, sentenceB, lasteventA, firsteventB, lasteventB)
                    generated_sentence = commongen_sentence(combinedContextAndIntent, min(32, maxLen+generationLengthSlack))
                    candidates.append(generated_sentence)

    return candidates

def getInvertedSentenceCandidatesFromCommonGen(sentenceA, sentenceB):
    candidates = []
    tagsA = getPosTags(sentenceA)
    tagsB = getPosTags(sentenceB)
    wordsA = getCentralWords(sentenceA)
    wordsB = getCentralWords(sentenceB)
    verbAndNNInA = [x[0] for x in filter(lambda x: x[0] in wordsA and 'VB' in x[1] or 'NN' in x[1], tagsA)][:numVerbsAndNN]
    verbAndNNInA = ' '.join(verbAndNNInA)
    verbAndNNInB = [x[0] for x in filter(lambda x: x[0] in wordsB and'VB' in x[1] or 'NN' in x[1], tagsB)][:numVerbsAndNN]
    verbAndNNInB = ' '.join(verbAndNNInB)
    
    # if VB and NN present, use them. Else generate sentences generically
    if verbAndNNInA and verbAndNNInB:
        intentA = getInvertedIntents(verbAndNNInA)
        intentB = getInvertedIntents(verbAndNNInB)
        sentences = generateSentencesFromVBAndNNIntents(intentA, intentB, ' '.join(wordsA), ' '.join(wordsB), tagsA, tagsB, len(sentenceB))
        candidates.extend(sentences)
    else:
        wordsA = getCentralWords(sentenceA)
        wordsB = getCentralWords(sentenceB)
        context = []
        for words in wordsB:
            processedSentenceContext = words
            intents = getInvertedIntents(words)
            sentences = generateSentencesFromIntents(intents, context, sentenceB, len(sentenceA))
            candidates.extend(sentences)

    return candidates


# commongen (t5 model) generation
# candidate file format - SentenceA (query) __eou__ SentenceB (response) followed by candidates in next line
# empty line separator between conversations

def generatedCommonGenCandidatesAndWriteToFile(conv_list, output_file='generated_sentences_commongen.txt', error_file='commongen_error_file.txt'):
    generatedResults = []
    errors = []
    i = 1
    for sentenceA, sentenceB in conv_list:
        expandedSentenceA = expandContactions(sentenceA)
        expandedSentenceB = expandContactions(sentenceB)
        print(i)
        i += 1
        try:
            results = getInvertedSentenceCandidatesFromCommonGen(expandedSentenceA, expandedSentenceB)
            generatedResults.append('{}\n{}'.format('{} __eou__ {}'.format(sentenceA, sentenceB), '\n'.join(results)))
        except Exception as e:
            errors.append('{}\n{}'.format(sentenceA, sentenceB))
            print(e)

    with open(output_file, 'w')as f:
        f.write('\n\n'.join(generatedResults))
        
    with open(error_file, 'w')as f:
        f.write('\n\n'.join(errors))


if __name__ == '__main__':   
	generatedCommonGenCandidatesAndWriteToFile(conv_list)