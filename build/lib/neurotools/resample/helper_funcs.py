import numpy as np

def triangle_area(v1, v2, v3):
    
    a = distance_squared_3D(v1, v2)
    b = distance_squared_3D(v2, v3)
    c = distance_squared_3D(v3, v1)
    
    area = .25 * np.sqrt(np.abs(4.0 * a * c - (a-b+c) * (a-b+c)))
    return area

def distance_squared_3D(p1, p2):
    
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    
    distSQ = (dx*dx) + (dy*dy) + (dz*dz)
    
    return np.float128(distSQ)

def get_coord_vec_length(coord):
    return np.sqrt(coord.dot(coord))

def flatten(collection):

    for item in collection:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

def normalize_vector(vector, return_norm=False):

    norm = get_coord_vec_length(vector)

    if return_norm:
        return vector / norm, norm

    return vector / norm
