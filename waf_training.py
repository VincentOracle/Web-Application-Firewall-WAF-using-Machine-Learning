import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import random
import joblib


def banner():
    print('''
*******************************************************************************************
*                                                                                         *
*   Web Application Firewall using Machine Learning and Features Engineering              *
*                                                                                         *
*                                                                                         *
*******************************************************************************************
    ''')

def usage():
    print('''
*******************************************************************************************
*                                                                                         *
*   Usage:                                                                                *
*   \t-d: dataset   (1 for HTTPParams, 2 for CSIC, 3 for both)                            *
*   \t-a: algorithm (1 for Support Vector Machine)                                        *
*   \t-t: test percentage (value from 0.05 to 0.95 .. Default is 0.2)                      *
*                                                                                         *
*   example: python waf_training.py -d 1 -a 1 -t 0.2                                       *
*                                                                                         *
*******************************************************************************************
    ''')

def split_csv(filename):
    global test_percentage
    global train_percentage
    dataset_file = open(filename+'.csv', 'r')
    li = dataset_file.readlines()
    header, rest = li[0], li[1:]
    test_num = int(len(rest) * test_percentage)
    train_num = int(len(rest) * train_percentage)
    dataset_file.close()
    random.shuffle(rest)
    dataset_file2 = open(filename+"_shuffled.csv", "w")
    new_lines = [header]+rest
    dataset_file2.writelines(new_lines)
    dataset_file2.close()
    
    dataset_file3 = open(filename+"_shuffled.csv", "r")
    lines = dataset_file3.readlines()
    test_records = lines[1:test_num]
    train_records = lines[test_num+1:]
    dataset_file3.close()

    test_csv = open(filename+"_test.csv", "w")
    new_test_lines = [header]+test_records
    test_csv.writelines(new_test_lines)

    train_csv = open(filename+"_train.csv", "w")
    new_train_lines = [header]+train_records
    train_csv.writelines(new_train_lines)

def process_dataset(option):
    option = int(option)
    if option == 1:
        split_csv("HTTPParams")
    elif option == 2:
        split_csv("CSIC")
    elif option == 3:
        split_csv("CSIC_HTTPParams")
    else:
        return False
    return True

algorithm = None
test_percentage = 0.2
train_percentage = 0.8
dataset = 1  # Default dataset option
algorithm_option = 1  # Default algorithm option

def choose_algorithm(option):
    global algorithm
    option = int(option)
    if option == 1:
        print('[+] \t Algorithm : Support Vector Machine (SVM)')
        from sklearn import svm
        algorithm = svm.SVC(kernel='linear', cache_size=7000)
    else:
        return False
    return True

def choose_dataset(option):
    global test_dataset
    global train_dataset
    option = int(option)
    if option == 1:
        print('[+] \t Dataset : HTTPParams 2015')
        process_dataset(option)
        dataset_file = 'HTTPParams'
        col_names = ['payloads', 'payload_len', 'alpha', 'non_alpha', 'attack_feature', 'label']
    elif option == 2:
        print('[+] \t Dataset : CSIC 2010')
        process_dataset(option)
        dataset_file = 'CSIC'
        col_names = ['method', 'url', 'payloads', 'payload_len', 'alpha', 'non_alpha', 'attack_feature', 'label']
    elif option == 3:
        print('[+] \t Dataset : CSIC 2010 & HTTPParams 2015')
        process_dataset(option)
        dataset_file = 'CSIC_HTTPParams'
        col_names = ['payload_len', 'alpha', 'non_alpha', 'attack_feature', 'label']
    else:
        return False

    test_dataset = pd.read_csv(dataset_file+'_test.csv', header=None, names=col_names, skiprows=1)
    train_dataset = pd.read_csv(dataset_file+'_train.csv', header=None, names=col_names, skiprows=1)

    return True

def train(ds, al, test, train):
    check_if_ds_selected = choose_dataset(ds)
    check_if_al_selected = choose_algorithm(al)

    if float(test) < 0.05 or float(test) > 0.95:
        print('Check your input by reading the usage below!')
        usage()
        exit()

    if float(train) < 0.05 or float(train) > 0.95:
        print('Check your input by reading the usage below!')
        usage()
        exit()

    if not check_if_ds_selected or not check_if_al_selected:
        print('Check your input by reading the usage below!')
        usage()
        exit()

    global algorithm
    global test_dataset
    global train_dataset
    feature_cols = ['payload_len', 'alpha', 'non_alpha', 'attack_feature']

    X1 = test_dataset[feature_cols]
    Y1 = test_dataset.label
    X2 = train_dataset[feature_cols]
    Y2 = train_dataset.label
    algorithm.fit(X2, Y2)
    Y2_pred = algorithm.predict(X1)
    algorithm.fit(X2, Y2)

    print('[+] \t Classification accuracy: {:.2f}%'.format(metrics.accuracy_score(Y1, Y2_pred) * 100))
    print('[+] \t Percentage of Anomaly requests in test set: {:.2f}%'.format(Y1.mean() * 100))
    print('[+] \t Percentage of Normal requests in test set: {:.2f}%'.format((1 - Y1.mean()) * 100))

    confusion = metrics.confusion_matrix(Y1, Y2_pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    print('[+] \t TP: {}, Total: {}'.format(TP, TP + TN + FP + FN))
    print('    \t TN: {}, Total: {}'.format(TN, TP + TN + FP + FN))
    print('    \t FP: {}, Total: {}'.format(FP, TP + TN + FP + FN))
    print('    \t FN: {}, Total: {}'.format(FN, TP + TN + FP + FN))

    print('\n[+] \t Metrics:')
    print('\t[-]  Accuracy Score (train_test_split): {:.2f}%'.format(metrics.accuracy_score(Y1, Y2_pred) * 100))
    print('\t[-]  Accuracy Score (k-fold): {:.2f}%'.format(cross_val_score(algorithm, X2, Y2, cv=100, scoring='accuracy').mean() * 100))
    print('\t[-]  Classification Error: {:.2f}%'.format((1 - metrics.accuracy_score(Y1, Y2_pred)) * 100))
    print('\t[-]  Sensitivity (Recall): {:.2f}%'.format(TP / float(TP + FN) * 100))
    print('\t[-]  Specificity: {:.2f}%'.format(TN / float(TN + FP) * 100))
    print('\t[-]  False Positive Rate: {:.2f}%'.format(FP / float(TN + FP) * 100))
    print('\t[-]  Precision: {:.2f}%'.format(TP / float(TP + FP) * 100))

    # Save the trained model
    joblib.dump(algorithm, 'waf_model.pkl')


def main():
    banner()

    if len(sys.argv) > 1:
        print('[-] \t WARNING: Running without command-line options. Using default values.')

    train(dataset, algorithm_option, test_percentage, train_percentage)


if __name__ == "__main__":
    main()
