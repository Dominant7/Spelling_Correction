from nltk import sent_tokenize


def LMDataConvert():
    import nltk
    from nltk.corpus import reuters
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import os

    # ������
    print(reuters.categories())
    
    # ��ȡ�����ĵ����ļ�ID�б�
    fileids = reuters.fileids()

    # ��ȡѵ�����Ͳ��Լ����ļ�ID�б�
    train_fileids = [fileid for fileid in fileids if fileid.startswith('training')]
    test_fileids = [fileid for fileid in fileids if fileid.startswith('test')]

    # ����һ��Ŀ¼�����洦������ı��ļ�
    test_dir = 'LM\\'
    train_dir = 'LM\\'
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    # ��ȡӢ��ͣ�ô��б�
    stop_words = set(stopwords.words('english'))

    # �����ļ�
    with open('LM\\test.txt', 'w', encoding='utf-8') as f:
        pass
    with open('LM\\train.txt', 'w', encoding='utf-8') as f:
        pass

    # ����ÿ���ļ��������浽�ı��ļ���
    for fileid in test_fileids:
        # ��ȡ�ı�����
        text = reuters.raw(fileid)
        # �־�
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # �ִ�
            tokens = word_tokenize(sentence)
            # ȥ��ͣ�ôʲ����ӳ��ַ���
            filtered_text = ' '.join(word for word in tokens if word.lower() not in stop_words)
            # ���洦�����ı����ļ�
            output_file = os.path.join(test_dir, 'test.txt')
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(filtered_text + '\n')
            
    for fileid in train_fileids:
        # ��ȡ�ı�����
        text = reuters.raw(fileid)
        # �־�
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # �ִ�
            tokens = word_tokenize(sentence)
            # ȥ��ͣ�ôʲ����ӳ��ַ���
            filtered_text = ' '.join(word for word in tokens if word.lower() not in stop_words)
            # ���洦�����ı����ļ�
            output_file = os.path.join(train_dir, 'train.txt')
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(filtered_text + '\n')

def LMTrain():
    import subprocess
    # �����ʻ��
    subprocess.run(['SRILM\\For PC\\ngram-count', '-text', 'LM\\train.txt', '-order', '3', '-write', 'LM\\count.txt'])
    # ѵ������ģ��
    subprocess.run(['SRILM\\For PC\\ngram-count', '-read', 'LM\\count.txt', '-order', '3', '-lm', 'LM\\train_model.lm', '-interpolate', '-kndiscount'])
#    subprocess.run(['SRILM\\For PC\\ngram', '-ppl', 'LM\\test_result.txt', '-order', '3', '-lm', 'LM\\train_model.lm', '>', 'LM\\test_result.ppl'])
    # ��������Ȳ������������ļ�
    with open('LM\\test_result.ppl', 'w') as out_file:
        subprocess.run(['SRILM\\For PC\\ngram', '-lm', 'LM\\train_model.lm', '-ppl', 'LM\\test.txt', '-order', '3'], stdout=out_file)