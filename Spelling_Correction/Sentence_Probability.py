import subprocess
def getSentenceProb(sentence, LMPath='LM\\train_model.lm', vocabPath='vocab.txt', order='3'):
    # 使用ngram命令计算句子的概率
    cmd = [
        'SRILM\\For PC\\ngram.exe',
        '-ppl', '-',  # 从标准输入读取句子
        '-lm', LMPath,  # 语言模型文件
        '-vocab', vocabPath,  # 词汇表文件
        '-order', order, # 阶数
        '-debug', '2'  # 输出详细信息
    ]
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(sentence + '\n')
#    print(stdout)
    log_prob = None
    for line in stdout.split('\n'):
        if 'logprob=' in line:
            log_prob = float(line.split('logprob=')[1].split()[0])
            break
    # print(sentence + '\nProb :' + str(log_prob))
    return log_prob, stdout

import re
import heapq

def extractProbabilities(output):
    # 正则表达式匹配 p( word | context ) = [source] prob [ logprob ]
    pattern = re.compile(r"p\(\s*(\w+|\<unk\>)\s*\|\s*.*?\s*\)\s*=\s*\[.*?\]\s*([\d\.e-]+)\s*\[.*?\]")
    words = []
    probs = []
    
    # for match in pattern.finditer(output):
    #     word, prob = match.groups()
    #     words.append(word)
    #     probs.append(float(prob))
    
    for match in pattern.finditer(output):
            word, prob = match.groups()
            prob_value = float(prob)
            if prob_value == float(0):
                words.append(word)
                probs.append(float('inf')) # -inf为oov，应排除oov位置
            else:
                words.append(word)
                probs.append(prob_value)
    return words, probs

def findSmallestProbWords(output, n):
    words, probs = extractProbabilities(output)
    smallestIndices = heapq.nsmallest(n, range(len(probs)), probs.__getitem__)
    return [(words[i], probs[i], i) for i in smallestIndices]