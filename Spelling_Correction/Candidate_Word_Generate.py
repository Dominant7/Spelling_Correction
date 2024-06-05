import nltk
import math

# 生成所有编辑距离为1的单词(不含分词与合词)
def edit_distance_1(word, uniqueChars, confusion_matrices, countDict):
    probDict = {}
    letters    = uniqueChars
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    # 生成时delete等于输错时insert
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
            delProb.append(math.log(confusion_matrices['ins'][L[-1], R[0]]) - math.log(countDict[L[-1]])) # 以对数存储概率
        else:
            pass
    
    for L, R in splits:
        if len(R) > 1:
            transposes.append(L + R[1] + R[0] + R[2:])
            transProb.append(math.log(confusion_matrices['trans'][R[1], R[0]]) - math.log(countDict[R[1] +  R[0]]))
        else:
            pass
        
    for L, R in splits:
        if R:
            for c in letters:
                replaces.append(L + c + R[1:])
                subProb.append(math.log(confusion_matrices['sub'][c, R[0]]) - math.log(countDict[c]))
        else:
            pass
    
    # 同理，生成insert等于delete
    for L, R in splits:
        for c in letters:
            inserts.append(L + c + R)
            insProb.append(math.log(confusion_matrices['del'][L[-1], c]) - math.log(countDict[L[-1] + c]))
    # 此处返回概率为对数
    return deletes + transposes + replaces + inserts, delProb + transProb + subProb + insProb

# 返回距离为1和2的字典
def edit_distance_2(word, uniqueChars, confusion_matrices, countDict):
    e1, e1Prob = edit_distance_1(word, uniqueChars, confusion_matrices, countDict)
    e2 = []
    e2Prob = []
    i = 0
    for e in e1:
        e2_temp, e2Prob_temp = edit_distance_1(e)
        e2.append(e2_temp)
        e2Prob_temp =  [a + e1Prob[i] for a in e2Prob_temp] # 对数概率，故为加法
        e2Prob.append(e2Prob_temp)
        i += 1
    e2.append(e1)
    e2Prob.append(e1Prob)
    return dict(zip(e2, e2Prob))
#    return (e2 for e1 in edit_distance_1(word) for e2 in edit_distance_1(e1))

# 返回在词汇表中的单词词典
def wordsInVocab(wordsDict, vocabulary):
    filteredDict = {}
    for key in vocabulary:
        if key in wordsDict:
            filteredDict[key] = wordsDict[key]
        elif set(nltk.word_tokenize(key)).issubset(vocabulary):
            filteredDict[key] = wordsDict[key]
        else:
            pass
    return filteredDict

# 生成分词后两部分都为词的单词组

# 生成融合两部分后得到的为词的单词组

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
        # 真词错误
        candidate.append({word:math.log(1 - epsilon)})
    else:
        # 非词错误
        pass
    return candidate