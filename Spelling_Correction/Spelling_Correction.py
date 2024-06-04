import torch
import numpy
import nltk
import LM_Generate
import Confusion_Matrix
import Candidate_Word_Generate

#LM_Generate.LMDataConvert()
#LM_Generate.LMTrain()
uniqueChars = Confusion_Matrix.createConfusionMatrix('count_1edit.txt', 'ConfusionMatrix.csv')

