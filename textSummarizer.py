import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline
import pandas as pd
from rouge_score import rouge_scorer
from statistics import mean

traincorpus = pd.read_csv('train.csv')
# testcorpus = pd.read_csv('test.csv')
# print(traincorpus.size) # 861339 rows

freqTable = dict()   # dictionary to hold frequency of words
# Tokenizing the text and counting frequency of words in the training corpus
for i in range(100): #I didn't do all the rows because my computer crashes
    text = traincorpus['article'][i]
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()] #taking out punctuation

    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
        freqTable['UNK'] = 1

def calculateScores(corpus, numArticles):
    precisions = []
    recalls = []

    #calculating the score of each sentence
    for i in range(numArticles):
        text = corpus['article'][i]
        sentences = sent_tokenize(text)
        sentenceValue = dict()

        for sentence in sentences:
            wordcount = 0
            for word, freq in freqTable.items():
                if word in sentence.lower():
                    wordcount += 1
                    if sentence in sentenceValue:
                        sentenceValue[sentence] += freq
                    else:
                        sentenceValue[sentence] = freq
                else:
                    continue
            if wordcount > 5:
                sentenceValue[sentence] = sentenceValue[sentence] / wordcount
            elif wordcount > 0:
                sentenceValue[sentence] = sentenceValue[sentence] / (wordcount * 2) #devaluing sentences with less than 5 words, mostly to remove strings about when article was updated or author names
            else:
                sentenceValue[sentence] = 0
            # print(sentence)
            # print(sentenceValue[sentence])

        # getting the average sentence value of the article
        sumValues = 0
        for sentence in sentenceValue:
            sumValues += sentenceValue[sentence]
        # print(sumValues)

        average = float(sumValues / len(sentenceValue))
        # print(average)

        #generating the summary
        summary = ''
        for sentence in sentences:
            if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.3 * average)):
                summary += " " + sentence
        # print(summary)

        #scoring the summary vs the human generated one
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = scorer.score(summary, corpus['highlights'][i])
        # print(type(scores.values()))
        for value in scores.values():
            precisions.append(value[0])
            recalls.append(value[1])

    #averaging the scores
    avgPrecision = mean(precisions)
    avgRecall = mean(recalls)
    return avgPrecision, avgRecall

testcorpus = pd.read_csv('test.csv')
# print(testcorpus.size) #34470 rows

#testing for best threshold
# print(calculateScores(traincorpus, 50,1.1))
# print(calculateScores(traincorpus, 50,1.2))
# print(calculateScores(traincorpus, 50,1.3))
# print(calculateScores(traincorpus, 50,1.4))
# print(calculateScores(traincorpus, 50,1.5))
# print(calculateScores(traincorpus, 50,1.6))
# (0.6196761111481983, 0.1532006217753826)
# (0.5457758095631468, 0.1778466639951163)
# (0.47190724717266713, 0.1959801913626315)
# (0.3863069835106179, 0.20547270519777072)
# (0.33017131546040235, 0.2146837389011829)
# (0.23876746589896208, 0.21210004668004964)


# print(calculateScores(traincorpus, 50))
# print(calculateScores(testcorpus, 50))
# print(calculateScores(testcorpus, 100))
# print(calculateScores(testcorpus, 150))
# print(calculateScores(testcorpus, 200))
