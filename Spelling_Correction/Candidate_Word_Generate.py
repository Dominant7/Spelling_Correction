import nltk

# �������б༭����Ϊ1�ĵ���(�����ִ���ϴ�)
def edit_distance_1(word, uniqueChars, confusion_matrices, countDict):
    probDict = {}
    letters    = uniqueChars
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
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
            delProb.append(confusion_matrices['ins'][L[-1], R[0]] / countDict[L[-1]])
        else:
            pass
    
    for L, R in splits:
        if len(R) > 1:
            transposes.append(L + R[1] + R[0] + R[2:])
            transProb.append(confusion_matrices['trans'][R[1], R[0]] / countDict[R[1] +  R[0]])
        else:
            pass
        
    for L, R in splits:
        if R:
            for c in letters:
                replaces.append(L + c + R[1:])
                subProb.append(confusion_matrices['sub'][c, R[0]] / countDict[c])
        else:
            pass
    
    # ͬ������insert����delete
    for L, R in splits:
        for c in letters:
            inserts.append(L + c + R)
            insProb.append(confusion_matrices['del'][L[-1], c] / countDict[L[-1] + c])
    
    return deletes + transposes + replaces + inserts, delProb + transProb + subProb + insProb

# ���ؾ���Ϊ1��2���ֵ�
def edit_distance_2(word, uniqueChars, confusion_matrices, countDict):
    e1, e1Prob = edit_distance_1(word, uniqueChars, confusion_matrices, countDict)
    e2 = []
    e2Prob = []
    i = 0
    for e in e1:
        e2_temp, e2Prob_temp = edit_distance_1(e)
        e2.append(e2_temp)
        e2Prob_temp =  [a * e1Prob[i] for a in e2Prob_temp]
        e2Prob.append(e2Prob_temp)
        i += 1
    e2.append(e1)
    e2Prob.append(e1Prob)
    return dict(zip(e2, e2Prob))
#    return (e2 for e1 in edit_distance_1(word) for e2 in edit_distance_1(e1))

# �����ڴʻ���еĵ��ʴʵ�
def wordsInVocab(wordsDict, vocabulary):
    filteredDict = {}
    for key in vocabulary:
        if key in wordsDict:
            filteredDict[key] = wordsDict[key]
    return filteredDict

# ���ɷִʺ������ֶ�Ϊ�ʵĵ�����

# �����ں������ֺ�õ���Ϊ�ʵĵ�����

def generateCandidate(word, uniqueChars, confusion_matrices, epsilon, countDict):
    vocab = []
    candidate = {}
    with open('vocab.txt', 'r', encoding='utf-8') as f:
        for line in f:
            vocabWord = line.strip()
            vocab.append(vocabWord)
    wordsDict = edit_distance_2(word, uniqueChars, confusion_matrices, countDict)
    candidate = wordsInVocab(wordsDict, vocab)
    if word in vocab:
        # ��ʴ���
        candidate.append({word:1 - epsilon})
    else:
        # �Ǵʴ���
        pass
    return candidate