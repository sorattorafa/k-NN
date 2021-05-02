def min_max(attribute): 
    return (attribute - attribute.min())/(attribute.max() - attribute.min())

def z_score(attribute):
    return (attribute - attribute.mean())/attribute.std()