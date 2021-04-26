from scipy.spatial import distance

def euclidean_distance(x,y):
    return distance.euclidean(x,y)

def manhattan_distance(x,y):
    return distance.cityblock(x,y)