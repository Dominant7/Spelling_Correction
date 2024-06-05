from nltk.metrics import confusionmatrix
import pandas as pd
import numpy as np
from collections import defaultdict
import csv

def readConfusionData(filename):
    confusion_dict = {
        'del': {},
        'ins': {},
        'sub': {},
        'trans': {}
    }
    
    count_dict = defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            pair, count = line.strip().split('\t')
            count = int(count)
            if '|' in pair:
                x, y = pair.split('|')
                x, y = x.lower(), y.lower()
                # if len(x) == 1 and len(y) == 1:
                #     confusion_dict['sub'][(y, x)] = count
                # elif len(x) == 2 and len(y) == 1:
                #     confusion_dict['ins'][(y, x[1])] = count
                # elif len(x) == 1 and len(y) == 2:
                #     confusion_dict['del'][(x, y[1])] = count
                # elif len(x) == 2 and len(y) == 2:
                #     confusion_dict['trans'][(x[1], y[1])] = count
                if len(x) == 1 and len(y) == 1:
                    confusion_dict['sub'][(x, y)] = count
                elif len(x) == 2 and len(y) == 1:
                    confusion_dict['ins'][(x, y)] = count
                elif len(x) == 1 and len(y) == 2:
                    confusion_dict['del'][(x, y)] = count
                elif len(x) == 2 and len(y) == 2:
                    confusion_dict['trans'][(x, y)] = count
                # x为错误项，y为正确项（存疑）
                count_dict[y] += count
    return confusion_dict, count_dict;

def generateConfusionMatrix(confusion_dict):
    unique_chars = set()
    for op in confusion_dict:
        for pair in confusion_dict[op]:
            unique_chars.update(pair[0])
            unique_chars.update(pair[1])
    unique_chars = sorted(unique_chars)
    char_index = {char: idx for idx, char in enumerate(unique_chars)}

    size = len(unique_chars)
    confusion_matrices = {
        'del': np.zeros((size, size), dtype=int),
        'ins': np.zeros((size, size), dtype=int),
        'sub': np.zeros((size, size), dtype=int),
        'trans': np.zeros((size, size), dtype=int),
    }

    for op in confusion_dict:
        for (x, y), count in confusion_dict[op].items():
            if op == 'sub':
                i, j = char_index[y], char_index[x]
                confusion_matrices[op][i, j] = count
            elif op == 'ins':
                i, j = char_index[y], char_index[x[1]]
                confusion_matrices[op][i, j] = count
            elif op == 'del':
                i, j = char_index[x], char_index[y[1]]
                confusion_matrices[op][i, j] = count
            elif op == 'trans':
                i, j = char_index[x[1]], char_index[y[1]]
                confusion_matrices[op][i, j] = count

    return confusion_matrices, unique_chars

def saveConfusionMatrices(matrices, chars, output_prefix, count_dict):
    for op in matrices:
        df = pd.DataFrame(matrices[op], index=chars, columns=chars)
        df.to_csv(f'{output_prefix}_{op}.csv')
    with open(f'{output_prefix}_count.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(count_dict.keys())
        writer.writerow(count_dict.values())

def createConfusionMatrix(confusionDataPath, confusionDataOutputPath):
    # Read the confusion data from the file
    confusionDict, countDict = readConfusionData(confusionDataPath)

    # Generate the confusion matrix and the list of characters
    confusionMatrix, uniqueChars = generateConfusionMatrix(confusionDict)

    # Save the confusion matrix to a CSV file for easy visualization
    saveConfusionMatrices(confusionMatrix, uniqueChars, confusionDataOutputPath, countDict)

    # Print the confusion matrix and unique characters (optional)
    print("Unique characters:", uniqueChars)
    return uniqueChars

def readConfusionMatrix(confusionDataPath='./ConfusionMatrix'):
    # 读取特定操作类型的混淆矩阵CSV文件
    opType = ['del', 'ins', 'sub', 'trans', 'count']
    confusionMatrix = {}
    for op in opType:
        if op == 'count':
            #confusionMatrix['count'] = (pd.read_csv(f'{confusionDataPath}_count.csv', header=None)).to_dict()
            df = pd.read_csv(f'{confusionDataPath}_count.csv', header=None)
            confusionMatrix['count'] = dict(zip(df.iloc[0], df.iloc[1].astype(int)))
            confusionMatrixCountDefaultDict = defaultdict(int)
            for key, value in confusionMatrix['count'].items():
                confusionMatrixCountDefaultDict[key] = value
            confusionMatrix['count'] = confusionMatrixCountDefaultDict
        else:
            confusionMatrix[op] = (pd.read_csv(f'{confusionDataPath}_{op}.csv', index_col=0)).to_dict()
    return confusionMatrix
