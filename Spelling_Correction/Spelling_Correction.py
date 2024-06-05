import torch
import numpy
import nltk
import LM_Generate
import Confusion_Matrix
import Candidate_Word_Generate
import Sentence_Probability

# 宏变量
EPSILON = 0.005 # 总出错概率 
FILE_PATH = 'testdata.txt'
OUTPUT_FILE_PATH = 'ans.txt'

#LM_Generate.LMDataConvert()
#LM_Generate.LMTrain()
uniqueChars = Confusion_Matrix.createConfusionMatrix('count_1edit.txt', 'ConfusionMatrix.csv')
confusionMatrix = Confusion_Matrix.readConfusionMatrix()
errorNums = []
with open(FILE_PATH, mode='r', encoding='utf-8') as file, \
    open(OUTPUT_FILE_PATH, mode='w', encoding='utf-8') as outfile:
    for line in file:
        seriesNum, errorNum, rawSentence = line.strip().split('\t')
        errorNums.append(int(errorNum))
        # 分词
        words = nltk.word_tokenize(rawSentence)
        
        # 移除句尾标点符号
        if words and words[-1] in {'.', '!', '?'}:
            endPunctuation = words.pop()
        else:
            endPunctuation = ''
            
        i = 0 # 词序数
        bestWords = []
        bestWordsProb = []
        for word in words:
            candidateDict = Candidate_Word_Generate.generateCandidate(word, uniqueChars, confusionMatrix, EPSILON, confusionMatrix['count'])
            logProbs = []
            candidates = []
            for candidate in candidateDict:
                wordsCopy = words.copy()
                wordsCopy[i] = candidate
                sentence = ' '.join(wordsCopy).replace(" n't", "n't").replace(" '", "'").replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
                logP_x_w = candidateDict[candidate]
                logP_w = Sentence_Probability.getSentenceProb(sentence)
                logProbs.append(logP_x_w + logP_w) # 将候选词概率存储
                candidates.append(candidate) # 将所有候选词存储
            maxValue = max(logProbs)
            maxIndex = logProbs.index(maxValue) # 找到最大概率值及其索引
            bestWords.append(candidates[maxIndex]) # 将最大概率值候选词加入
            bestWordsProb.append(maxValue) # 将最大概率加入
            i += 1
        # 根据错误数n选择n个概率最大的词(对数概率最大的n个值相加一定得到总和最大的值)
        # 将数组和对应的下标组合成一个元组列表
        indexArray = list(enumerate(bestWordsProb))
        sortedArray = sorted(indexArray, key=lambda x: x[1], reverse=True)
        
        # 获取最大的n个值的下标
        topIndices = [index for index, value in sortedArray[:errorNum]]
        for index in topIndices:
            words[index] = bestWords[index]
            
        processedSentence = ' '.join(words).replace(" n't", "n't").replace(" '", "'").replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
        # 还原句末标点
        if endPunctuation:
            processedSentence += endPunctuation
        # 写入输出文件    
        outfile.write(processedSentence + '\n')

