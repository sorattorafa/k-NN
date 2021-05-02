import sys
import numpy as np

def save_response(instances):
    output = open('responsess.txt','w')
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

def check_params():
    if len(sys.argv) < 3:
        print("Please, provide two files as arguments: training and test")
        sys.exit()
   
def load_training():
    training_matrix = None
    try:
        training_matrix = np.loadtxt(sys.argv[1], delimiter=' ')
    except OSError:
        print("File training does not exist")
    return training_matrix

def load_test():    
    test_matrix = None
    try:
        test_matrix = np.loadtxt(sys.argv[2], delimiter=' ')      
    except OSError:
        print("File test does not exist")
    return test_matrix  
