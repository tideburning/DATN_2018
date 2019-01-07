import pandas as pd
from collections import Counter
from prefixspan import PrefixSpan
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt     

PATH = 'data/CSV/'
NORM_TRAIN = PATH + 'Parsed_output_http_csic_2010_weka_with_duplications_RAW-RFC2616_escd_v02_norm.csv'
TEST_ANOM = PATH + 'Parsed_output_http_csic_2010_weka_with_duplications_RAW-RFC2616_escd_v02_anom.csv'
TEST_NORM = PATH + 'Parsed_output_http_csic_2010_weka_with_duplications_RAW-RFC2616_escd_v02_norm_test.csv'

'''
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
'''
def read_data(file):
    data = pd.read_csv(file, keep_default_na=False, low_memory=False)
    print("READ")
    print(str(len(data)))
    return data.values.tolist()

def preprocessing(file):
    print("PREPROCESSING...")
    data = read_data(file)
    freq = dict()
    for request in data:
        if request[2] not in freq:
            freq[request[2]] = []
        if request[16] != '':
            freq[request[2]].append(str(request[16]))
    print("PREPROCESSING DONE")
    return freq

def mining():

    train = preprocessing(NORM_TRAIN)
    mining = dict()
    print("MINING....")
    for d in train:
        data = []
        for i in range(len(train[d])//10):
            data.append(train[d][i*10:(i+1)*10])
        ps = PrefixSpan(data)
        mining[d] = ps.frequent(2, closed=True)
        #print(mining[d])  
    print("MINING DONE, PATTERN CREATED")
    return mining

def matchingAnorm(data, rule):
    if len(rule) == 0:
        return False
    for _rule in rule:
        count = 0
        for r in _rule[1]:
            if r in data:
                count+=1
        if count/len(data)<0:
            return True
    return False

def matchingNorm(data, rule):
    if len(rule) == 0:
        return False
    for _rule in rule:
        count = 0
        for r in _rule[1]:
            if r in data:
                count+=1
        if count/len(data)>0:
            return True
    return False

def print_matrix(cm):
    plt.clf()
    plt.imshow(cm)
    classNames = ['Abnormal','Normal']
    plt.title('CONFUSION MATRIX')
    plt.ylabel('PREDICTED')
    plt.xlabel('ACTUAL')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks,classNames)
    plt.yticks(tick_marks,classNames)
    s=[['TN','FP'],['FN','TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i,str(s[i][j]) + "=" + str(cm[i][j]))
    plt.savefig("Confusion_matrix_0.0.png")
    plt.show()

def test(rule, anom, norm):
    sum = 0
    pred = []
    true = []
#    print (len(anom) // 10 + len(norm) // 10)
    for d in anom:
        true = true + [0] * (len(anom[d])//10)
        for i in range(len(anom[d])//10):
            data = anom[d][i*10:(i+1)*10]
            if (d in rule) and matchingAnorm(data, rule[d]):
                pred.append(0)
            else:
                pred.append(1)
    
    for d in norm:
        true = true + [1] * (len(norm[d])//10)
        for i in range(len(norm[d])//10):
            data = norm[d][i*10:(i+1)*10]
            if (d in rule) and matchingNorm(data, rule[d]):
                pred.append(0)
            else:
                pred.append(1)    

#    print (len(true), len(pred))
    matrix = confusion_matrix(true, pred)
#    print("CONFUSION_MATRIX")
    print_matrix(matrix)
#    print(str(matrix))
    TP, FP = matrix[0]
    FN, TN = matrix[1]
    Precision = (TP * 1.0) / (TP + FP)
    Recall = (TP * 1.0) / (TP + FN)
    ACC = ((TP + TN) * 1.0) / (TP + TN + FP + FN)
    F1 = 2.0 * Precision * Recall / (Precision + Recall)
    print("PRECISION: ", Precision*100, "%")
    print("RECALL: ", Recall*100 , "%")
    print("ACCURACY: ", ACC *100 , "%")
    print("F1-SCORE: ", F1*100 , "%")

    return F1


def print_patterns(rule):
    w = open("log_rules_0.3.txt", "w")
    for data in rule:
        w.write(data)
        for line in rule[data]:
            for element in line:
                    w.write(str(element))
            w.write("\n")
        w.write("\n\n\n")
    w.close()

def print_preprocessing(data):
    w = open("preprocessing_train.txt", "w")
    for _data in data:
        w.write(_data)
        for line in data[_data]:
            for element in line:
                    w.write(str(element))
            w.write("\n")
        w.write("\n\n\n")
    w.close()

def main():
    patterns = mining()
    # print(rule)
    test_anom_data = preprocessing(TEST_ANOM)
    test_norm_data = preprocessing(TEST_NORM)
    F1 = test(patterns, test_anom_data, test_norm_data)

#    print("F1-SCORE = ", F1)
#    print_patterns(patterns)
if __name__ == '__main__':
    main()
