# -*- coding: utf-8 -*-
import nltk
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import os
import math

rootDir = os.path.abspath('.') + '\\ICML'

Stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))
englishwords_set = set(nltk.corpus.words.words())
allWords_set = set([])
papers = {}
word_idf = {}
fileCounts = 0
output_files = []

for parent, dirnames, filenames in os.walk(rootDir):

    for filename in filenames:

        if parent == (rootDir + '\\7. Kernel Methods'):
            if filename != 'desktop.ini':
                output_files.append(filename)

        fileCounts += 1

        f = open(os.path.join(parent, filename), 'r', encoding='utf-8')
        paper = f.read()

        wordlist = nltk.word_tokenize(paper)

        words_withoutStops = []

        for word in wordlist:
            if Stemmer.stem(word.lower()) in englishwords_set:
                if word.lower() not in stop_words:
                    words_withoutStops.append(Stemmer.stem(word.lower()))

        papers[filename] = words_withoutStops
        word_set = set(words_withoutStops)

        for word in word_set:
            if word in word_idf:
                word_idf[word] += 1
            else:
                word_idf[word] = 1

        allWords_set = allWords_set | word_set

        f.close()

for (word, idf) in word_idf.items():
    word_idf[word] = math.log10(fileCounts/idf)

allWords_list = list(allWords_set)
allWords_list = sorted(allWords_list)
allwords_file = open('assignment1\\allwords.txt', 'w+', encoding='utf-8')
print(allWords_list, file=allwords_file)

result_file = open('assignment1\\result.txt', 'w+', encoding='utf-8')

for (filename, wordlist) in papers.items():

    word_tf = {}
    result = {}
    wordCount = 0
    for word in wordlist:
        wordCount += 1
        if word in word_tf:
            word_tf[word] += 1
        else:
            word_tf[word] = 1

    for (word, tf) in word_tf.items():
        tf = tf / wordCount
        result[word] = tf * word_idf[word]

    if filename in set(output_files):
        count = 0
        print('[', end="", file=result_file)
        for i in range(1, len(allWords_list)+1):
            if allWords_list[i-1] in result:
                count += 1
                if count == len(result):
                    print('%5d:%.10f' % (i, result[allWords_list[i-1]]), end="", file=result_file)
                    break
                print('%5d:%.10f,' % (i, result[allWords_list[i-1]]), end="", file=result_file)

        print(']', file=result_file)

allwords_file.close()
result_file.close()

