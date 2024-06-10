'''

import numpy
import nltk
import LM_Generate
import Confusion_Matrix
import Candidate_Word_Generate
import Sentence_Probability

# 宏变量
EPSILON = 0.05 # 总出错概率 
FILE_PATH = 'testdata.txt'
OUTPUT_FILE_PATH = 'result.txt'
VOCABULARY_PATH = 'vocab.txt'
REAL_WORD_DETECTION = False # 是否检测真词错误

#LM_Generate.LMDataConvert()
#LM_Generate.LMTrain()
uniqueChars = Confusion_Matrix.createConfusionMatrix('count_1edit.txt', 'ConfusionMatrix')
uniqueChars.remove('0')
confusionMatrix = Confusion_Matrix.readConfusionMatrix()
errorNums = []
vocab = [] # 全为小写
PUNCTUATION_LIST = [',', '!', '?', ':', ';', '(', ')', '\'', '\"',  '[', ']']

def getCandidateWithProb(words, detectRealWordError=False):
        i = 0 # 词序数
        bestWords = []
        bestWordsProb = []
        for word in words:
            if word in PUNCTUATION_LIST: # 分词得到的标点符号进行生成词会出现问题
                candidateDict = {}
            else:
                candidateDict = Candidate_Word_Generate.generateCandidate(word, uniqueChars, confusionMatrix, EPSILON, vocab, confusionMatrix['count'], detectRealWordError)
            logProbs = []
            candidates = []
            if len(candidateDict) == 0: # 不检测真词时返回为空
                i += 1
                bestWords.append('$') # 占位用，防止索引错乱
                bestWordsProb.append(float('-inf')) # 概率取最小防止选到
            else:
                for candidate in candidateDict:
                    wordsCopy = words.copy()
                    wordsCopy[i] = candidate
                    sentence = ' '.join(wordsCopy).replace(" n't", "n't").replace(" '", "'").replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
                    logP_x_w = candidateDict[candidate]
                    logP_w = Sentence_Probability.getSentenceProb(sentence)
                    logProbs.append(logP_x_w + logP_w) # 将候选词概率存储
                    candidates.append(candidate) # 将所有候选词存储
                maxValue = max(logProbs) # 最大概率候选词
                maxIndex = logProbs.index(maxValue) # 找到最大概率值及其索引
                bestWords.append(candidates[maxIndex]) # 将最大概率值候选词加入
                bestWordsProb.append(maxValue) # 将最大概率加入
                i += 1
        # 根据错误数n选择n个概率最大的词(对数概率最大的n个值相加一定得到总和最大的值)
        # 将数组和对应的下标组合成一个元组列表
        indexArray = list(enumerate(bestWordsProb))
        sortedArray = sorted(indexArray, key=lambda x: x[1], reverse=True)
        return sortedArray, bestWords



with open(VOCABULARY_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            vocabWord = line.strip()
            vocab.append(vocabWord.lower())
with open(FILE_PATH, mode='r', encoding='utf-8') as file, \
    open(OUTPUT_FILE_PATH, mode='w', encoding='utf-8') as outfile:
    for line in file:
        seriesNum, errorNum, rawSentence = line.strip().split('\t')
        errorNum = int(errorNum)
        errorNums.append(errorNum)
        # 分词
        words = nltk.word_tokenize(rawSentence)
        
        # 移除句尾标点符号
        if words and words[-1] in {'.', '!', '?'}:
            endPunctuation = words.pop()
        else:
            endPunctuation = ''
        
        sortedArray, bestWords = getCandidateWithProb(words)
       """ 
        i = 0 # 词序数
        bestWords = []
        bestWordsProb = []
        for word in words:
            candidateDict = Candidate_Word_Generate.generateCandidate(word, uniqueChars, confusionMatrix, EPSILON, vocab, confusionMatrix['count'])
            logProbs = []
            candidates = []
            if len(candidateDict) == 0: # 不检测真词时返回为空
                i += 1
                bestWords.append('$') # 占位用，防止索引错乱
                bestWordsProb.append(float('-inf')) # 概率取最小防止选到
            else:
                for candidate in candidateDict:
                    wordsCopy = words.copy()
                    wordsCopy[i] = candidate
                    sentence = ' '.join(wordsCopy).replace(" n't", "n't").replace(" '", "'").replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
                    logP_x_w = candidateDict[candidate]
                    logP_w = Sentence_Probability.getSentenceProb(sentence)
                    logProbs.append(logP_x_w + logP_w) # 将候选词概率存储
                    candidates.append(candidate) # 将所有候选词存储
                maxValue = max(logProbs) # 最大概率候选词
                maxIndex = logProbs.index(maxValue) # 找到最大概率值及其索引
                bestWords.append(candidates[maxIndex]) # 将最大概率值候选词加入
                bestWordsProb.append(maxValue) # 将最大概率加入
                i += 1
        # 根据错误数n选择n个概率最大的词(对数概率最大的n个值相加一定得到总和最大的值)
        # 将数组和对应的下标组合成一个元组列表
        indexArray = list(enumerate(bestWordsProb))
        sortedArray = sorted(indexArray, key=lambda x: x[1], reverse=True)
        """      
        # 获取最大的n个值的下标
        topIndices = [index for index, value in sortedArray[:errorNum]]
        # 真词错误
        realWordsErrorCount = int(0)
        for index in topIndices:
            if bestWords[index] == '$':
                # 说明有未检测的真词错误
                realWordsErrorCount += 1
            else:
                words[index] = bestWords[index]
        if realWordsErrorCount != 0 and REAL_WORD_DETECTION: 
            sortedArray, bestWords = getCandidateWithProb(words, True)
            topIndices = [index for index, value in sortedArray[:realWordsErrorCount]]
            for index in topIndices:
                words[index] = bestWords[index]
        else:
             pass
        processedSentence = ' '.join(words).replace(" n't", "n't").replace(" '", "'").replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
        # 还原句末标点
        if endPunctuation:
            processedSentence += endPunctuation
        # 写入输出文件    
        outfile.write(seriesNum + '\t' + processedSentence + '\n')

'''


import numpy
import nltk
import LM_Generate
import Confusion_Matrix
import Candidate_Word_Generate
import Sentence_Probability
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import freeze_support

# 宏变量
EPSILON = 0.05  # 总出错概率
FILE_PATH = 'testdata.txt'
OUTPUT_FILE_PATH = 'result.txt'
VOCABULARY_PATH = 'vocab.txt'
REAL_WORD_DETECTION = False  # 是否检测真词错误
EDIT_DISTANCE = 2 # 生成最大编辑距离

# LM_Generate.LMDataConvert(False)
# LM_Generate.LMTrain()
uniqueChars = Confusion_Matrix.createConfusionMatrix('count_1edit.txt', 'ConfusionMatrix')
uniqueChars.remove('0')
confusionMatrix = Confusion_Matrix.readConfusionMatrix()
errorNums = []
vocab = []  # 全为小写
PUNCTUATION_LIST = [',', '!', '?', ':', ';', '(', ')', '\'', '\"', '[', ']']


def getCandidateWithProb(words, detectRealWordError=False):
    i = 0  # 词序数
    bestWords = []
    bestWordsProb = []
    for word in words:
        if word in PUNCTUATION_LIST:  # 分词得到的标点符号进行生成词会出现问题
            candidateDict = {}
        else:
            candidateDict = Candidate_Word_Generate.generateCandidate(word, uniqueChars, confusionMatrix, EPSILON, vocab, confusionMatrix['count'], detectRealWordError, EDIT_DISTANCE)
        logProbs = []
        candidates = []
        if len(candidateDict) == 0:  # 不检测真词时返回为空
            i += 1
            bestWords.append('$')  # 占位用，防止索引错乱
            bestWordsProb.append(float('-inf'))  # 概率取最小防止选到
        else:
            for candidate in candidateDict:
                wordsCopy = words.copy()
                wordsCopy[i] = candidate
                sentence = ' '.join(wordsCopy).replace(" n't", "n't").replace(" '", "'").replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
                logP_x_w = candidateDict[candidate]
                logP_w, _ = Sentence_Probability.getSentenceProb(sentence)
                logProbs.append(logP_x_w + logP_w)  # 将候选词概率存储
                candidates.append(candidate)  # 将所有候选词存储
            maxValue = max(logProbs)  # 最大概率候选词
            maxIndex = logProbs.index(maxValue)  # 找到最大概率值及其索引
            if word[0].isupper(): # 首字母大写则还原
                bestWords.append(candidates[maxIndex].capitalize())  # 将最大概率值候选词加入
            else:
                bestWords.append(candidates[maxIndex])  # 将最大概率值候选词加入
            bestWordsProb.append(maxValue)  # 将最大概率加入
            i += 1
    # 根据错误数n选择n个概率最大的词(对数概率最大的n个值相加一定得到总和最大的值)
    # 将数组和对应的下标组合成一个元组列表
    indexArray = list(enumerate(bestWordsProb))
    sortedArray = sorted(indexArray, key=lambda x: x[1], reverse=True)
    return sortedArray, bestWords


def process_line(line):
    seriesNum, errorNum, rawSentence = line.strip().split('\t')
    print(seriesNum)
    errorNum = int(errorNum)
    errorNums.append(errorNum)
    # 分词
    words = nltk.word_tokenize(rawSentence)

    # 移除句尾标点符号
    if words and words[-1] in {'.', '!', '?'}:
        endPunctuation = words.pop()
    else:
        endPunctuation = ''

    sortedArray, bestWords = getCandidateWithProb(words)
    # 获取最大的n个值的下标
    topIndices = [index for index, value in sortedArray[:errorNum]]
    # 真词错误
    realWordsErrorCount = int(0)
    for index in topIndices:
        if bestWords[index] == '$':
            # 说明有未检测的真词错误
            realWordsErrorCount += 1
        else:
            words[index] = bestWords[index]
    if realWordsErrorCount != 0 and REAL_WORD_DETECTION:
        _, LMOutput = Sentence_Probability.getSentenceProb(rawSentence)
        smallestProbWords = Sentence_Probability.findSmallestProbWords(LMOutput, realWordsErrorCount)
        for word, prob, index in smallestProbWords:
            if word in PUNCTUATION_LIST:  # 分词得到的标点符号进行生成词会出现问题
                candidateDict = {}
            else:
                candidateDict = Candidate_Word_Generate.generateCandidate(words[index], uniqueChars, confusionMatrix, EPSILON, vocab, confusionMatrix['count'], True, EDIT_DISTANCE)
            if len(candidateDict) == 0:  # 不检测真词时返回为空
                continue
            logProbs = []
            candidates = []
            for candidate in candidateDict:
                wordsCopy = words.copy()
                wordsCopy[index] = candidate
                sentence = ' '.join(wordsCopy).replace(" n't", "n't").replace(" '", "'").replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
                logP_x_w = candidateDict[candidate]
                logP_w, _ = Sentence_Probability.getSentenceProb(sentence)
                # print(_)
                logProbs.append(logP_x_w + logP_w)  # 将候选词概率存储
                candidates.append(candidate)  # 将所有候选词存储
            maxValue = max(logProbs)  # 最大概率候选词
            maxIndex = logProbs.index(maxValue)  # 找到最大概率值及其索引
            if word[0].isupper(): # 首字母大写则还原
                words[index] = candidates[maxIndex].capitalize()  # 将最大概率值候选词加入
            else:
                words[index] = candidates[maxIndex]  # 将最大概率值候选词加入
        # sortedArray, bestWords = getCandidateWithProb(words, True)
        # topIndices = [index for index, value in sortedArray[:realWordsErrorCount]]
        # for index in topIndices:
        #     words[index] = bestWords[index]
    else:
        pass
    processedSentence = ' '.join(words).replace(" n't", "n't").replace(" '", "'").replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    # 还原句末标点
    if endPunctuation:
        processedSentence += endPunctuation
    return seriesNum + '\t' + processedSentence + '\n'

if __name__ == '__main__':
    freeze_support()  

    # 读取词汇表
    with open(VOCABULARY_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            vocabWord = line.strip()
            vocab.append(vocabWord.lower())

    # 多线程处理文件
    with open(FILE_PATH, mode='r', encoding='utf-8') as file, \
            open(OUTPUT_FILE_PATH, mode='w', encoding='utf-8') as outfile, \
            ThreadPoolExecutor() as executor:
        results = list(executor.map(process_line, file))
        outfile.writelines(results)
