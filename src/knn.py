import numpy as np
import pandas as pd
from statistics import mode
import distances as distances

class KNeighbors:
    def __init__(self, n_neighbors=3, trainig_data=None, distance_type='euclidean'):
        self.n_neighbors = n_neighbors
        condition = distance_type == 'manhattan'
        self.calc_distance = distances.manhattan_distance if(condition) else distances.euclidean_distance        
        self.training_dataframe = trainig_data
            
    def classify(self,new_object):                
        nearest_neighbors = [[float('inf'),None] for _ in range(self.n_neighbors)] # create space to nearest_neighbors
        for _class, _object in self.training_dataframe.iterrows(): # _class = index and _object = data series
            neighbor_distance = self.calc_distance(new_object,_object.values) # get neighbor distance
            nearest_neighbors.sort(key=lambda dist: dist[0], reverse=True) # sort by smallest to largest
            if neighbor_distance < nearest_neighbors[0][0]: # if actual distance < smallest
                nearest_neighbors[0][0] = neighbor_distance # smallest = atual
                nearest_neighbors[0][1] = _class # smallest name = actual class name
        nearest_neighbors = [_class[1] for _class in nearest_neighbors] # set nearest_neighbors
        return mode(nearest_neighbors) # return nearest_neighbors into mode