import sys
import numpy as np
import pandas as pd
import knn as knn
import random

def normalize(matrix,normalization):
    dataframe = pd.DataFrame(matrix,index=matrix[:,-1].astype(int))
    labels_index = len(dataframe.columns)-1
    labels = dataframe.columns[labels_index]
    dataframe.drop(labels, axis=1, inplace=True)
    return dataframe.apply(normalization , axis=1).copy()

def reduce_dataframe(dataframe, percent):
    if(percent == 1):
        return dataframe
    dataframe_indexes = dataframe.index.values
    dataframe_indexes = list(dict.fromkeys(dataframe_indexes))
    data_class = []
    for _index in dataframe_indexes:
        data_class.append(dataframe.groupby(dataframe.index).get_group(_index).sample(frac=percent,random_state=random.randrange(0,100)))

    return pd.concat(data_class)

def get_confusion_matrix(actual_knn, training_norm_dataframe, test_norm_dataframe):   
    confusion_matrix = [ [ 0 for _ in range(10) ] for _ in range(10) ]
    for _class, _object in test_norm_dataframe.iterrows():
        classified_class = actual_knn.classify(_object.values)
        confusion_matrix[_class][classified_class] += 1

    return confusion_matrix

def test_matrices(test_matrix, training_matrix, instances_config):
    instances = []
    for percent in (instances_config['percents']):
        for normalization in (instances_config['normalizations']):
            test_norm_dataframe = normalize(test_matrix,normalization)
            training_norm_dataframe = reduce_dataframe(normalize(training_matrix,normalization),percent)
            for distance in (instances_config['distances']):
                for k in (instances_config['k']):
                    test_instance = {
                        'k': k,
                        'percent':percent,
                        'normalization': normalization.__name__,
                        'distance':distance,
                    }
                    actual_knn = knn.KNeighbors(n_neighbors=k, distance_type=distance, trainig_data =training_norm_dataframe)
                    confusion_matrix = get_confusion_matrix(actual_knn, training_norm_dataframe, test_norm_dataframe)
                    test_instance['confusion_matrix'] = confusion_matrix
                    instances.append(test_instance)
    return instances