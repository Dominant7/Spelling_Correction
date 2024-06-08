from nltk import sent_tokenize


def LMDataConvert():
    import nltk
    from nltk.corpus import reuters
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import os

    # 输出类别
    print(reuters.categories())
    
    # 获取所有文档的文件ID列表
    fileids = reuters.fileids()

    # 获取训练集和测试集的文件ID列表
    train_fileids = [fileid for fileid in fileids if fileid.startswith('training')]
    test_fileids = [fileid for fileid in fileids if fileid.startswith('test')]

    # 创建一个目录来保存处理过的文本文件
    test_dir = 'LM\\'
    train_dir = 'LM\\'
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    # 获取英语停用词列表
    stop_words = set(stopwords.words('english'))

    # 创建文件
    with open('LM\\test.txt', 'w', encoding='utf-8') as f:
        pass
    with open('LM\\train.txt', 'w', encoding='utf-8') as f:
        pass

    # 处理每个文件，并保存到文本文件中
    for fileid in test_fileids:
        # 获取文本内容
        text = reuters.raw(fileid)
        # 分句
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # 分词
            tokens = word_tokenize(sentence)
            # 去除停用词并连接成字符串
            filtered_text = ' '.join(word for word in tokens if word.lower() not in stop_words)
            # 保存处理后的文本到文件
            output_file = os.path.join(test_dir, 'test.txt')
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(filtered_text + '\n')
            
    for fileid in train_fileids:
        # 获取文本内容
        text = reuters.raw(fileid)
        # 分句
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # 分词
            tokens = word_tokenize(sentence)
            # 去除停用词并连接成字符串
            filtered_text = ' '.join(word for word in tokens if word.lower() not in stop_words)
            # 保存处理后的文本到文件
            output_file = os.path.join(train_dir, 'train.txt')
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(filtered_text + '\n')

def LMTrain():
    import subprocess
    # 构建词汇表
    subprocess.run(['SRILM\\For PC\\ngram-count', '-text', 'LM\\train.txt', '-order', '3', '-write', 'LM\\count.txt'])
    # 训练语言模型
    subprocess.run(['SRILM\\For PC\\ngram-count', '-read', 'LM\\count.txt', '-order', '3', '-lm', 'LM\\train_model.lm', '-interpolate', '-kndiscount'])
#    subprocess.run(['SRILM\\For PC\\ngram', '-ppl', 'LM\\test_result.txt', '-order', '3', '-lm', 'LM\\train_model.lm', '>', 'LM\\test_result.ppl'])
    # 计算困惑度并将结果输出到文件
    with open('LM\\test_result.ppl', 'w') as out_file:
        subprocess.run(['SRILM\\For PC\\ngram', '-lm', 'LM\\train_model.lm', '-ppl', 'LM\\test.txt', '-order', '3'], stdout=out_file)