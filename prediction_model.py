import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.utils import shuffle
from collections import Counter
import matplotlib.pyplot as plt
import statistics
from sklearn import datasets

def load_data(file, columns):
    data = pd.read_csv(file, names= columns)
    final_data = shuffle(data)
    return final_data

def normalise_train(data):
    dic = {}
    columns = data.columns[:-1]
    for i in columns:
        dic[i] = tuple([data[i].max(), data[i].min()])
        if (data[i].max() - data[i].min()) == 0:
            data[i] = (data[i] - data[i].min()) 
        else:
            data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())
    return data, dic     

def normalize_data(X_train: pd.DataFrame):
    minimum = X_train.min()
    denom = X_train.max() - minimum
    denom[denom == 0] = 1
    X_train = (X_train - minimum)/denom
    return minimum, denom, X_train

def normalise_test(data, max_min_data):
    columns = data.columns[:-1]
    for i in columns:
        if (max_min_data[i][0] - max_min_data[i][1]) == 0:
            data[i] = (data[i] - max_min_data[i][1]) 
        else:
            data[i] = (data[i] - max_min_data[i][1]) / (max_min_data[i][0] - max_min_data[i][1])
    return data       

def train_knn(train, test, k, features, label):
    train_data =  train[features]
    test_data = test[features]
    train_target = train[label]
    test_target = test[label]
    predictions = []
    for i in range(0, len(test_data)):
        lst = []
        for j in range(0, len(train_data)):
            dist = distance.euclidean(test_data.iloc[i],train_data.iloc[j])
            lst.append(tuple([dist, train_target.iloc[j]]))
        lst1 = sorted(lst, key = lambda x: x[0], reverse = False)[:k]

        class_name = []
        for i in range(0, len(lst1)):
            class_name.append(lst1[i][1])

        c = Counter(class_name)
        pred = c.most_common(1)
        predictions.append(pred[0][0])
    
    acc = 0
    for i in range(0, len(test_target)):
        
        if test_target.iloc[i] == predictions[i]:
            acc +=1

    accuracy = acc/ len(test_target) 

    return accuracy, predictions, test_target


def testing_model(file, columns):
    accuracy_training = []
    standard_deviation_training = []
    accuracy_testing = []
    standard_deviation_testing = []
    k_values = []
    accuracies_te = {}
    accuracies_tr ={}

    for i in range(0, 20):
        data = load_data(file, columns)
        train_initial , test_initial = train_test_split(data, test_size=0.2)
        train, max_min_data = normalise_train(train_initial)
        test = normalise_test(test_initial, max_min_data)
        
        for j in range(1, 52, 2):
            print(i)
            acc_test = train_knn(train, test, j)
            acc_train = train_knn(train, train, j)

            if j not in accuracies_te:
                accuracies_te[j] = []
                accuracies_te[j].append(acc_test)
            else:
                accuracies_te[j].append(acc_test)

            if j not in accuracies_tr:
                accuracies_tr[j] = []
                accuracies_tr[j].append(acc_train)
            else:
                accuracies_tr[j].append(acc_train)

    k_values = []
    for i in range(1, 52, 2):
        k_values.append(i)

    lst_test = list(accuracies_te.keys())
    lst_test.sort()
    a_testing = {z: accuracies_te[z] for z in lst_test}

    lst_train = list(accuracies_tr.keys())
    lst_train.sort()
    a_training = {x: accuracies_tr[x] for x in lst_train}


    for i in a_testing:
        mean_accuracy_test = sum(a_testing[i])/len(a_testing[i])
        std_dev_test = statistics.pstdev(a_testing[i])
        accuracy_testing.append(mean_accuracy_test)
        standard_deviation_testing.append(std_dev_test)

    for i in a_training:
        mean_accuracy_train = sum(a_training[i])/len(a_training[i])
        std_dev_train = statistics.pstdev(a_training[i])
        accuracy_training.append(mean_accuracy_train)
        standard_deviation_training.append(std_dev_train)
    
   
    return accuracy_training, standard_deviation_training, accuracy_testing, standard_deviation_testing, k_values
        
    
def graph_plot(accuracy, standard_deviation, k_values, flag):
    if flag == False:
        plt.errorbar(k_values, accuracy, yerr = standard_deviation, marker = 'o')
        plt.xlabel("value of K")
        plt.ylabel("Accuracy over training data")
        plt.show()
    elif flag == True:
        plt.errorbar(k_values, accuracy, yerr = standard_deviation, marker = 'o')
        plt.xlabel("value of K")
        plt.ylabel("Accuracy over testing data")
        plt.show()

def plot(accuracy, F1score, k_values):
    plt.plot(k_values, accuracy)
    plt.xlabel("value of K")
    plt.ylabel("Accuracy")
    plt.show()
    
    plt.plot(k_values, F1score)
    plt.xlabel("value of K")
    plt.ylabel("F1 score")
    plt.show()

def k_fold(data, k, label, total_labels):
    #finding percentage of each label in the dataset
    percentage = {}
    for i in total_labels:
        percentage[i] = len(data[data[label] == i]) / len(data)

    #finding the length of each fold
    length_fold = int(len(data) / k)

    #finding total of each label in folds
    total_class_numbers = {}
    for i in percentage:
        total_class_numbers[i] = round(percentage[i] * length_fold)

    
    k_fold_data = []
    temp_data = data
    for i in range(1, k):
        lst_data = []
        for j in total_class_numbers:
            temp = temp_data[temp_data[label] == j]
            temp = temp.sample(n= total_class_numbers[j],replace= False)
            index_lst = list(temp.index)
            temp_data = temp_data.drop(index_lst)

            lst_data.append(temp)
        d = pd.concat(lst_data)
        k_fold_data.append(shuffle(d))


    new_lst = []
    for j in total_class_numbers:
        temp = temp_data[temp_data[label] == j]
        temp = shuffle(temp)
        temp = temp[:total_class_numbers[j]]
        index_lst = list(temp.index)
        temp_data = temp_data.drop(index_lst)
        new_lst.append(temp)
    
    d = pd.concat(new_lst)
    k_fold_data.append(shuffle(d))
    
    return k_fold_data

def calculate_performance_two_class(predictions, target, total_labels):
    true_predictions = {}
    count = 0
    for i in total_labels:
        true_predictions[i] = {}

    for i in true_predictions:
        for j in total_labels:
                true_predictions[i][j] = 0

    for i in range(0, len(predictions)):
        if predictions[i] == target.iloc[i]:
                count += 1
                true_predictions[predictions[i]][predictions[i]] += 1
        
        else:
                true_predictions[target.iloc[i]][predictions[i]] += 1

    accuracy = count / (len(predictions))

    precision = true_predictions[1][1] / (true_predictions[1][1] + true_predictions[0][1])

    recall = true_predictions[1][1] / (true_predictions[1][1] + true_predictions[1][0])

    F1_score = (2*(precision*recall))/(precision + recall)

    return accuracy, precision, recall, F1_score

def convert_to_df(X, y):
    df = pd.DataFrame(X)
    df['target']= y
    return df

def calculate_performance_multiple_class(predictions, test_data_label_check, total_labels):
    true_predictions = {}
    for i in total_labels:
        true_predictions[i] = {}

    for i in true_predictions:
        for j in total_labels:
                true_predictions[i][j] = 0

    for i in range(0, len(predictions)):
        if predictions[i] == test_data_label_check.iloc[i]:
                true_predictions[predictions[i]][predictions[i]] += 1
        
        else:
                true_predictions[test_data_label_check.iloc[i]][predictions[i]] += 1
    
    accuracy = 0
    for i in true_predictions:
        accuracy += true_predictions[i][i]
    
    accuracy = accuracy/ len(predictions)

    precision = 0
    for i in true_predictions:
        total_count_precision = 0
        for j in true_predictions:
            total_count_precision += true_predictions[j][i]
        if total_count_precision != 0:
            precision += true_predictions[i][i] / total_count_precision

    final_precision = precision / len(true_predictions)

    recall = 0
    for i in true_predictions:
        total_count_recall = sum(true_predictions[i].values())
        recall += true_predictions[i][i] / total_count_recall

    final_recall = recall / len(true_predictions)

    F1_score = (2*(final_precision*final_recall))/(final_precision + final_recall)
     
    return accuracy, final_precision, final_recall, F1_score


def main(file):   
    if file == "Diabetes":
        digits = datasets.load_digits(return_X_y=True)
        data = convert_to_df(digits[0], digits[1])
        attributes = list(data.columns[:-1])
        label = data.columns[-1]

        total_labels = data[label].unique()
        attribute_type = {}
        k_fold_data = k_fold(data, 10, label, total_labels)

   
        k_values = [1, 3, 5]
        
        accuracy_final = []
        F1score_final = []
        for j in k_values:
            final_accuracy = []
            final_precision = []
            final_recall = []
            final_F1score = []
            for i in range(0, len(k_fold_data)):
                print(i)
                train_d =  k_fold_data[:i] + k_fold_data[i+1:]
                tr_data = pd.concat(train_d)
                te_data  =  k_fold_data[i]
                train_dummy = tr_data.copy()
                test_dummy  = te_data.copy()

                train, max_min_data = normalise_train(train_dummy)
                test = normalise_test(test_dummy, max_min_data)

                acc_test, predictions, test_label = train_knn(train, test, j, attributes, label)
                
                accuracy, precision, recall, F1_score = calculate_performance_multiple_class(predictions, test_label, total_labels)

                final_accuracy.append(accuracy)
                final_precision.append(precision)
                final_recall.append(recall)
                final_F1score.append(F1_score)

            accuracy_f  = sum(final_accuracy)/len(final_accuracy)
            precision_f = sum(final_precision)/len(final_precision)
            recall_f   = sum(final_recall)/len(final_recall)
            F1_score_f = sum(final_F1score)/len(final_F1score)

            accuracy_final.append(accuracy_f)
            F1score_final.append(F1_score_f)

            
        print(accuracy_final)
        print(F1score_final)
        plot(accuracy_final, F1score_final, k_values)

if __name__ == '__main__':
    file = "Diabetes"
    main(file)