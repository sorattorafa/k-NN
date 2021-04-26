import utils as utils

if __name__ == '__main__':   
    utils.check_params()
    test_matrix = utils.load_test()
    training_matrix = utils.load_training()
    instances =  utils.create_instances(test_matrix, training_matrix)
    utils.save_response(instances)
    print('Finish!')