import torch
import numpy
import nltk
import LM_Generate
import Confusion_Matrix
import Candidate_Word_Generate
import Sentence_Probability

# �����
EPSILON = 0.005 # �ܳ������ 
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
        # �ִ�
        words = nltk.word_tokenize(rawSentence)
        
        # �Ƴ���β������
        if words and words[-1] in {'.', '!', '?'}:
            endPunctuation = words.pop()
        else:
            endPunctuation = ''
            
        i = 0 # ������
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
                logProbs.append(logP_x_w + logP_w) # ����ѡ�ʸ��ʴ洢
                candidates.append(candidate) # �����к�ѡ�ʴ洢
            maxValue = max(logProbs)
            maxIndex = logProbs.index(maxValue) # �ҵ�������ֵ��������
            bestWords.append(candidates[maxIndex]) # ��������ֵ��ѡ�ʼ���
            bestWordsProb.append(maxValue) # �������ʼ���
            i += 1
        # ���ݴ�����nѡ��n���������Ĵ�(������������n��ֵ���һ���õ��ܺ�����ֵ)
        # ������Ͷ�Ӧ���±���ϳ�һ��Ԫ���б�
        indexArray = list(enumerate(bestWordsProb))
        sortedArray = sorted(indexArray, key=lambda x: x[1], reverse=True)
        
        # ��ȡ����n��ֵ���±�
        topIndices = [index for index, value in sortedArray[:errorNum]]
        for index in topIndices:
            words[index] = bestWords[index]
            
        processedSentence = ' '.join(words).replace(" n't", "n't").replace(" '", "'").replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
        # ��ԭ��ĩ���
        if endPunctuation:
            processedSentence += endPunctuation
        # д������ļ�    
        outfile.write(processedSentence + '\n')

