import nltk
import math

def logSmoothed(num, epsilon=1e-6):
    return math.log(num + epsilon)

# 生成所有编辑距离为1的单词(不含分词与合词)
def edit_distance_1(word, uniqueChars, confusion_matrices, countDict):
    probDict = {}
    letters    = uniqueChars
    splits     = [(word[:i].lower(), word[i:].lower()) for i in range(len(word) + 1)]
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
            if L:
                delProb.append(logSmoothed(confusion_matrices['ins'][L[-1]][R[0]]) - logSmoothed(countDict[L[-1]])) # 以对数存储概率
            else:
                delProb.append(logSmoothed(confusion_matrices['ins']['>'][R[0]]) - logSmoothed(countDict['>'])) # 句子开头为>(存疑)
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
    
    # 同理，生成insert等于delete
    for L, R in splits:
        for c in letters:
            inserts.append(L + c + R)
            if L:
                insProb.append(logSmoothed(confusion_matrices['del'][L[-1]][c]) - logSmoothed(countDict[L[-1] + c]))
            else:
                insProb.append(logSmoothed(confusion_matrices['del']['>'][c]) - logSmoothed(countDict['>' + c])) # 句子开头为>(存疑)
    # 此处返回概率为对数
    return deletes + transposes + replaces + inserts, delProb + transProb + subProb + insProb

# 返回距离为1和2的字典
def edit_distance_2(word, uniqueChars, confusion_matrices, countDict):
    e1, e1Prob = edit_distance_1(word, uniqueChars, confusion_matrices, countDict)
    e2 = []
    e2Prob = []
    i = 0
    for e in e1:
        e2_temp, e2Prob_temp = edit_distance_1(e, uniqueChars, confusion_matrices, countDict)
        e2.extend(e2_temp)
        e2Prob_temp =  [a + e1Prob[i] for a in e2Prob_temp] # 对数概率，故为加法
        e2Prob.extend(e2Prob_temp)
        i += 1
    e2.extend(e1)
    e2Prob.extend(e1Prob)
    return dict(zip(e2, e2Prob))
#    return (e2 for e1 in edit_distance_1(word) for e2 in edit_distance_1(e1))

# 返回在词汇表中的单词词典
def wordsInVocab(wordsDict, vocabulary):
    filteredDict = {}
    for key in wordsDict:
        if key in vocabulary:
            filteredDict[key] = wordsDict[key]
#        elif set(nltk.word_tokenize(key)).issubset(vocabulary): # 这个改成自己写，nltk分词太多了，只需要保留空格分词即可
        elif set(key.split(' ')).issubset(vocabulary):
            filteredDict[key] = wordsDict[key]
        else:
            pass
    return filteredDict

# 生成分词后两部分都为词的单词组

# 生成融合两部分后得到的为词的单词组

def generateCandidate(word, uniqueChars, confusion_matrices, epsilon, vocab, countDict, detectRealWordError=False):
    candidate = {}
    if detectRealWordError: # 加个检测首字母大写
        #    wordsDict = edit_distance_2(word, uniqueChars, confusion_matrices, countDict)
        e1, e1Prob = edit_distance_1(word, uniqueChars, confusion_matrices, countDict) # 试一试编辑距离1
        wordsDict = dict(zip(e1, e1Prob))
        candidate = wordsInVocab(wordsDict, vocab)
        if word.lower() in vocab:
            # 真词错误
            candidate.update({word:logSmoothed(1 - epsilon)})
        else:
            # 非词错误
            pass
    else:
        if word.lower() in vocab:
            pass
        else:
            e1, e1Prob = edit_distance_1(word, uniqueChars, confusion_matrices, countDict)
            wordsDict = dict(zip(e1, e1Prob))
            candidate = wordsInVocab(wordsDict, vocab)
    return candidate