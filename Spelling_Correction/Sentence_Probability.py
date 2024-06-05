import subprocess
def getSentenceProb(sentence, LMPath='LM\\train_model.lm', vocabPath='vocab.txt', order=3):
    # ʹ��ngram���������ӵĸ���
    cmd = [
        'ngram',
        '-ppl', '-',  # �ӱ�׼�����ȡ����
        '-lm', LMPath,  # ����ģ���ļ�
        '-vocab', vocabPath,  # �ʻ���ļ�
        '-order', order, # ����
        '-debug', '2'  # �����ϸ��Ϣ
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