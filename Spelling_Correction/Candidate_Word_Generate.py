import nltk
import math

def logSmoothed(num, epsilon=1e-6):
    return math.log(num + epsilon)

# �������б༭����Ϊ1�ĵ���(�����ִ���ϴ�)
def edit_distance_1(word, uniqueChars, confusion_matrices, countDict):
    probDict = {}
    letters    = uniqueChars
    splits     = [(word[:i].lower(), word[i:].lower()) for i in range(len(word) + 1)]
    # ����ʱdelete�������ʱinsert
    deletes = []
    delProb = []
    transposes = []
    transProb = []
    replaces = []
    subProb = []
    inserts = []
    insProb = []
    for L, R in splits:
        if R:
            deletes.append(L + R[1:])
            if L:
                delProb.append(logSmoothed(confusion_matrices['ins'][L[-1]][R[0]]) - logSmoothed(countDict[L[-1]])) # �Զ����洢����
            else:
                delProb.append(logSmoothed(confusion_matrices['ins']['>'][R[0]]) - logSmoothed(countDict['>'])) # ���ӿ�ͷΪ>(����)
        else:
            pass
    
    for L, R in splits:
        if len(R) > 1:
            transposes.append(L + R[1] + R[0] + R[2:])
            transProb.append(logSmoothed(confusion_matrices['trans'][R[1]][R[0]]) - logSmoothed(countDict[R[1] +  R[0]]))
        else:
            pass
        
    for L, R in splits:
        if R:
            for c in letters:
                replaces.append(L + c + R[1:])
                subProb.append(logSmoothed(confusion_matrices['sub'][c][R[0]]) - logSmoothed(countDict[c]))
        else:
            pass
    
    # ͬ������insert����delete
    for L, R in splits:
        for c in letters:
            inserts.append(L + c + R)
            if L:
                insProb.append(logSmoothed(confusion_matrices['del'][L[-1]][c]) - logSmoothed(countDict[L[-1] + c]))
            else:
                insProb.append(logSmoothed(confusion_matrices['del']['>'][c]) - logSmoothed(countDict['>' + c])) # ���ӿ�ͷΪ>(����)
    # �˴����ظ���Ϊ����
    return deletes + transposes + replaces + inserts, delProb + transProb + subProb + insProb

# ���ؾ���Ϊ1��2���ֵ�
def edit_distance_2(word, uniqueChars, confusion_matrices, countDict):
    e1, e1Prob = edit_distance_1(word, uniqueChars, confusion_matrices, countDict)
    e2 = []
    e2Prob = []
    i = 0
    for e in e1:
        e2_temp, e2Prob_temp = edit_distance_1(e, uniqueChars, confusion_matrices, countDict)
        e2.extend(e2_temp)
        e2Prob_temp =  [a + e1Prob[i] for a in e2Prob_temp] # �������ʣ���Ϊ�ӷ�
        e2Prob.extend(e2Prob_temp)
        i += 1
    e2.extend(e1)
    e2Prob.extend(e1Prob)
    return dict(zip(e2, e2Prob))
#    return (e2 for e1 in edit_distance_1(word) for e2 in edit_distance_1(e1))

# �����ڴʻ���еĵ��ʴʵ�
def wordsInVocab(wordsDict, vocabulary):
    filteredDict = {}
    for key in wordsDict:
        if key in vocabulary:
            filteredDict[key] = wordsDict[key]
#        elif set(nltk.word_tokenize(key)).issubset(vocabulary): # ����ĳ��Լ�д��nltk�ִ�̫���ˣ�ֻ��Ҫ�����ո�ִʼ���
        elif set(key.split(' ')).issubset(vocabulary):
            filteredDict[key] = wordsDict[key]
        else:
            pass
    return filteredDict

# ���ɷִʺ������ֶ�Ϊ�ʵĵ�����

# �����ں������ֺ�õ���Ϊ�ʵĵ�����

def generateCandidate(word, uniqueChars, confusion_matrices, epsilon, vocab, countDict, detectRealWordError=False):
    candidate = {}
    if detectRealWordError: # �Ӹ��������ĸ��д
        #    wordsDict = edit_distance_2(word, uniqueChars, confusion_matrices, countDict)
        e1, e1Prob = edit_distance_1(word, uniqueChars, confusion_matrices, countDict) # ��һ�Ա༭����1
        wordsDict = dict(zip(e1, e1Prob))
        candidate = wordsInVocab(wordsDict, vocab)
        if word.lower() in vocab:
            # ��ʴ���
            candidate.update({word:logSmoothed(1 - epsilon)})
        else:
            # �Ǵʴ���
            pass
    else:
        if word.lower() in vocab:
            pass
        else:
            e1, e1Prob = edit_distance_1(word, uniqueChars, confusion_matrices, countDict)
            wordsDict = dict(zip(e1, e1Prob))
            candidate = wordsInVocab(wordsDict, vocab)
    return candidate