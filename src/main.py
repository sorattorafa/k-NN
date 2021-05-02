import dataframes as dataframes
import middlewares as middlewares
from scores import z_score, min_max

if __name__ == '__main__':   
    middlewares.check_params()
    test_matrix = middlewares.load_test()
    training_matrix = middlewares.load_training()
    instances_config = {
        'percents' : [0.25,0.5,1],
        'normalizations' : [min_max,z_score],
        'distances' : ['euclidean','manhattan'],
        'k' : [1,3,5,7,9,11,13,15,17,19]
    }
    instances =  dataframes.test_matrices(test_matrix, training_matrix, instances_config)
    middlewares.save_response(instances)
    print('Finish!')