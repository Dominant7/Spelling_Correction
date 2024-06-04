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
                count_dict[x] += count
                count_dict[x + y] += count

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
    print("Confusion matrix:")
    print(confusionMatrix)
    return uniqueChars