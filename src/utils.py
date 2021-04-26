import sys
import numpy as np
import pandas as pd
import knn as knn
import random

def min_max(attribute): 
    return (attribute - attribute.min())/(attribute.max() - attribute.min())

def z_score(attribute):
    return (attribute - attribute.mean())/attribute.std()

def normalize(matrix,normalization):
    dataframe = pd.DataFrame(matrix,index=matrix[:,-1].astype(int))
    labels_index = len(dataframe.columns)-1
    labels = dataframe.columns[labels_index]
    dataframe.drop(labels, axis=1, inplace=True)
    return dataframe.apply(normalization , axis=1).copy()

def save_response(instances):
    output = open('response.txt','w')
    for instance in instances:
        output.write("------------------------------------------------------------------------------\n")
        output.write(f"| K = {instance['k']} | Percent: {instance['percent']*100} % | Normalization: {instance['normalization']} | Distance: {instance['distance']} |\n\n")
        cf = np.asarray(instance['confusion_matrix'])
        accuracy = (cf.trace()/cf.sum())*100
        output.write(f"| Hit rate: {accuracy} %, error rate: {'{:.2f}'.format(100 - accuracy)} % |\n\n")
        output.write(f"confusion_matrix: \n")      
        for _class in instance['confusion_matrix']:
            output.write(f"{_class}\n")        
        output.write("\n----------------------------------------------------------------------------\n")
        
def reduce_dataframe(dataframe, percent):
    if(percent == 1):
        return dataframe
    dataframe_indexes = dataframe.index.values
    dataframe_indexes = list(dict.fromkeys(dataframe_indexes))
    data_class = []
    for _index in dataframe_indexes:
        data_class.append(dataframe.groupby(dataframe.index).get_group(_index).sample(frac=percent,random_state=random.randrange(0,100)))

    return pd.concat(data_class)

def check_params():
    if len(sys.argv) < 3:
        print("Please, provide two files as arguments: training and test")
        sys.exit()
   
def load_training():
    try:
        training_matrix = np.loadtxt(sys.argv[1], delimiter=' ')
    except OSError:
        print("File training does not exist")
    return training_matrix

def load_test():    
    try:
        test_matrix = np.loadtxt(sys.argv[2], delimiter=' ')        
    except OSError:
        print("File test does not exist")   
    return test_matrix

def get_instance(test_instance, training_norm_dataframe, test_norm_dataframe):
    k = test_instance['k']
    distance = test_instance['distance']
    actual_knn = knn.KNeighbors(n_neighbors=k, distance_type=distance, trainig_data =training_norm_dataframe)    
    confusion_matrix = [ [ 0 for _ in range(10) ] for _ in range(10) ]
    
    for _class, _object in test_norm_dataframe.iterrows():
        classified_class = actual_knn.classify(_object.values)
        confusion_matrix[_class][classified_class] += 1

    return confusion_matrix

def create_instances(test_matrix, training_matrix):
    instances = []
    
    instances_config = {
        'percents' : [0.25,0.5,1],
        'normalizations' : [min_max,z_score],
        'distances' : ['euclidean','manhattan'],
        'k' : [1,3,5,7,9,11,13,15,17,19]
    }

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
                    confusion_matrix = get_instance(test_instance, training_norm_dataframe, test_norm_dataframe)
                    test_instance['confusion_matrix'] = confusion_matrix
                    instances.append(test_instance)
    return instances