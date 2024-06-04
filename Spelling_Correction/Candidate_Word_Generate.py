import nltk

# 生成所有编辑距离为1的单词(不含分词与合词)
def edit_distance_1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces   = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts    = [L + c + R for L, R in splits for c in letters + '-']
    return set(deletes + transposes + replaces + inserts)

def edit_distance_2(word):
    return (e2 for e1 in edit_distance_1(word) for e2 in edit_distance_1(e1))

# 查找词汇表中与目标词编辑距离为1的所有单词
def find_candidates(word, vocabulary):
    candidates = edit_distance_1(word)
    return candidates.intersection(vocabulary)

# 生成分词后两部分都为词的单词组

# 生成融合两部分后得到的为词的单词组

def generateCandidate(word):
    vocab = []
    candidate = []
    with open('vocab.txt', 'r', encoding='utf-8') as f:
        for line in f:
            vocabWord = line.strip()
            vocab.append(vocabWord)
    if word in vocab:
        # 真词错误
        candidate = find_candidates(word, vocab) + word
    else:
        # 非词错误
        candidate = find_candidates(word, vocab)
    return candidate