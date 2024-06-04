import nltk

# �������б༭����Ϊ1�ĵ���(�����ִ���ϴ�)
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

# ���Ҵʻ������Ŀ��ʱ༭����Ϊ1�����е���
def find_candidates(word, vocabulary):
    candidates = edit_distance_1(word)
    return candidates.intersection(vocabulary)

# ���ɷִʺ������ֶ�Ϊ�ʵĵ�����

# �����ں������ֺ�õ���Ϊ�ʵĵ�����

def generateCandidate(word):
    vocab = []
    candidate = []
    with open('vocab.txt', 'r', encoding='utf-8') as f:
        for line in f:
            vocabWord = line.strip()
            vocab.append(vocabWord)
    if word in vocab:
        # ��ʴ���
        candidate = find_candidates(word, vocab) + word
    else:
        # �Ǵʴ���
        candidate = find_candidates(word, vocab)
    return candidate