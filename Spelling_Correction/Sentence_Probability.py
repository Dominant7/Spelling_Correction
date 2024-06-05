import subprocess
def getSentenceProb(sentence, LMPath='LM\\train_model.lm', vocabPath='vocab.txt', order=3):
    # 使用ngram命令计算句子的概率
    cmd = [
        'ngram',
        '-ppl', '-',  # 从标准输入读取句子
        '-lm', LMPath,  # 语言模型文件
        '-vocab', vocabPath,  # 词汇表文件
        '-order', order, # 阶数
        '-debug', '2'  # 输出详细信息
    ]
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(sentence + '\n')
    
    log_prob = None
    for line in stdout.split('\n'):
        if 'logprob=' in line:
            log_prob = float(line.split('logprob=')[1].split()[0])
            break
        else:
            print('Error in getting sentence probability!')
    return log_prob