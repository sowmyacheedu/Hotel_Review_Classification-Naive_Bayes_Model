# SeekTruth.py : Classify text objects into two categories
#
#Names - userID
#Sowmya Cheedu - scheedu
#Adarsha Reddy pedda gorla - adpeddag
#Vamsidhar pagadala - vpagada
#
#
# Based on skeleton code by D. Crandall, October 2021
#
import math
import re
import sys
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


def load_file(filename):
    objects = []
    labels = []
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ', 1)
            labels.append(parsed[0] if len(parsed) > 0 else "")
            objects.append(parsed[1] if len(parsed) > 1 else "")
    # print({"objects": objects, "labels": labels, "classes": list(set(labels))})

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}


# classifier : Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# should return a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated classlabel for test_data["objects"][i]
#
# Do not change the return type or parameters of this function!
#
def classifier(train_data, test_data):
    # print(train_data)
    truthCount = 0
    decepCount = 0
    for i in train_data["labels"]:
        if i == 'truthful':
            truthCount += 1
        else:
            decepCount += 1

    totalTruthProbab = truthCount / (truthCount + decepCount)
    totalDecepProbab = decepCount / (truthCount + decepCount)
    # print(truthCount, decepCount)
    # print(totalTruthProbab, totalDecepProbab)

    # df = pd.DataFrame(train_data["objects"])
    # df=pd.read_csv('deceptive.train.txt', sep=" " )
    # print("shape",df.shape)
    # print(df)

    wordGivenTruthProbabMap = dict()
    totaltruthWordCount = 0
    truthWordCountMap = dict()
    decepWordCountMap = dict()
    wordGivenDecepProbabMap = dict()
    totalDecepDisWordCount = 0
    wordCountMap = dict()
    count = 0
    stop_words = ['hadn', 'from', "wasn't", 'other', 'both', 'so', 'yourselves', 'hers', 'or', 'up', 'again', 'now',
                  'having', 'himself', "you've", 'into', 'why', 'll', 'those', 'whom', 'nor', 'my', 'there', 'it',
                  'before', 'only', 'at', 'do', 'then', 'she', 'further', "doesn't", 'but', 'your', 'which', 'of',
                  "you'll", 'until', 'herself', 'some', 'own', 'yourself', 'couldn', 'over', 'them', 'on', 's', 'been',
                  'doing', 'i', 'by', "she's", 'her', 'wasn', "mightn't", 'doesn', 'themselves', 'is', 'if', 'where',
                  'between', 'about', 'y', 'd', 'needn', 'ourselves', 'didn', 'and', 'ma', 'aren', 'after', 'theirs',
                  'being', 'am', 'should', "that'll", 'that', 'yours', 'because', 'can', 'him', 'in', 'its', 't',
                  'through', 'm', 'off', 'these', 'they', 'have', "shouldn't", 'their', "haven't", "shan't", 'when',
                  'mustn', 'no', 'was', 'will', 'very', 'under', "weren't", 'than', "wouldn't", 'most', 're', 've',
                  'you', 'be', 'the', 'isn', 'such', 'who', 'too', "needn't", 'while', 'as', 'itself', "you're", 'hasn',
                  "didn't", "should've", 'all', "won't", 'an', 'ours', 'against', 'for', 'what', 'few', "couldn't",
                  "you'd", 'we', 'below', 'me', "it's", 'has', 'not', 'out', 'o', 'are', 'were', 'wouldn', 'weren', 'a',
                  'down', 'shan', "hadn't", "hasn't", 'once', 'mightn', 'same', 'with', 'during', 'just', "aren't",
                  'this', "isn't", 'does', 'to', 'don', 'haven', 'shouldn', "mustn't", 'his', 'had', 'each', 'myself',
                  'did', 'how', 'here', 'our', 'above', 'won', 'any', 'he', 'ain', "don't", 'more']
    # tokenizer = RegexpTokenizer(r'\w+')
    # lemmatizer = WordNetLemmatizer()
    # print(stop_words)
    # stop_words = ['a','the','is']
    for i in range(0, len(train_data['objects'])):
        # words = train_data['objects'][i].split(" ")
        # train_data['objects'][i] = re.sub("[,./'()!;*%:?&$<>#-]", " ", train_data["objects"][i])
        train_data['objects'][i] = re.sub('\W', ' ', train_data['objects'][i])
        # words = train_data['objects'][i].split(" ")
        # tokens = list(tokenizer.tokenize(train_data['objects'][i].lower()))
        tokens = train_data['objects'][i].lower().split()
        for j in tokens:
            # j = lemmatizer.lemmatize(j)
            # j = re.sub("[,./'!()-;*%:?&$<>#]", "", j)
            if j not in stop_words:
                if j == '' or j == ' ':
                    continue
                if j in wordCountMap:
                    wordCountMap[j] += 1
                else:
                    wordCountMap[j] = 1
                if train_data['labels'][i] == 'truthful':
                    totaltruthWordCount += 1
                    if j in truthWordCountMap:
                        truthWordCountMap[j] += 1
                    else:
                        truthWordCountMap[j] = 1
                else:
                    totalDecepDisWordCount += 1
                    if j in decepWordCountMap:
                        decepWordCountMap[j] += 1
                    else:
                        decepWordCountMap[j] = 1

    # print("totaltruthWordCount", totaltruthWordCount)
    # print("totalDecepDisWordCount", totalDecepDisWordCount)
    # print(truthWordCountMap)

    '''
    f = open("words.txt", "w+")
    for i in wordCountMap:
        f.write(i+'\n')
    f.close()
    '''

    missTCount = 0

    # add missing words in truth and decep
    for i in truthWordCountMap:
        if i not in decepWordCountMap:
            decepWordCountMap[i] = 0
            missTCount += 1
    missDCount = 0
    for i in decepWordCountMap:
        if i not in truthWordCountMap:
            truthWordCountMap[i] = 0
            missDCount += 1

    # print("Missing T in D", missTCount, "Missing D in T", missDCount)

    # alpha =1 and increment each count and recompute total count
    totaltruthWordCount = 0
    for i in truthWordCountMap:
        truthWordCountMap[i] += 1
        totaltruthWordCount += truthWordCountMap[i]
    totalDecepDisWordCount = 0
    for j in decepWordCountMap:
        decepWordCountMap[j] += 1
        totalDecepDisWordCount += decepWordCountMap[j]
    # print("totaltruthWordCount", totaltruthWordCount)
    # print("totalDecepDisWordCount", totalDecepDisWordCount)

    # Compute Likelihood
    # f = open("truthWords.txt", "w+")
    for i in truthWordCountMap:
        wordGivenTruthProbabMap[i] = truthWordCountMap[i] / totaltruthWordCount
        # f.write(i + "-" + str(wordGivenTruthProbabMap[i]) + '\n')
    # f.close()
    # f = open("decepWords.txt", "w+")
    for i in decepWordCountMap:
        wordGivenDecepProbabMap[i] = decepWordCountMap[i] / totalDecepDisWordCount
        # f.write(i + "-" + str(wordGivenDecepProbabMap[i]) + '\n')
    # f.close()

    # Now that stage is set,testing the model
    # f = open("missingwords.txt", "w+")
    result = []
    for i in range(0, len(test_data["objects"])):
        truthfulProbability = totalTruthProbab
        deceptiveProbability = totalDecepProbab
        test_data["objects"][i] = re.sub('\W', ' ', test_data["objects"][i])
        # tokenss = list(tokenizer.tokenize(test_data['objects'][i].lower()))
        tokenss = test_data["objects"][i].lower().split()
        for each in tokenss:
            # each = lemmatizer.lemmatize(each)
            # for each in wordss:
            # each = each.lower()
            if each == '' or each == ' ' or each in stop_words:
                continue

            # if each not in wordGivenTruthProbabMap or each not in wordGivenDecepProbabMap:
            # f.write(each + "\n")

            # TRUTH
            if each in wordGivenTruthProbabMap:
                truthfulProbability *= wordGivenTruthProbabMap[each]
            # DECEPTIVE
            if each in wordGivenDecepProbabMap:
                deceptiveProbability *= wordGivenDecepProbabMap[each]

        if truthfulProbability >= deceptiveProbability:
            result.append('truthful')
        else:
            result.append('deceptive')

    #f.close()
    # print(len(result))
    return result

    # This is just dummy code -- put yours here!
    # return [test_data["classes"][0]] * len(test_data["objects"])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if (sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    results = classifier(train_data, test_data_sanitized)

    # calculate accuracy
    correct_ct = sum([(results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"]))])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))
