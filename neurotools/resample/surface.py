import numpy as np
import nibabel as nib
from .distance_helpers import get_signed_distance_helper
from .surface_file import SurfaceFile

def make_barycentric_weights(from_s, to_s):
    
    # Init weights as list of mappings
    weights = [dict() for _ in range(len(to_s.coords))]
    
    # Get distance helper for from
    mySignedHelp = get_signed_distance_helper(from_s)
    
    # For each node in 
    for i, coord in enumerate(to_s.coords):
        
        # Get nodes + weights and fill into weights
        nodes, baryWeights = mySignedHelp.barycentricWeights(coord)

        for node, weight in zip(nodes, baryWeights):
            if weight != 0:
                weights[i][node] = weight

    return weights

def gen_weights_adap_bary_area(cur_sphere, new_sphere, curAreas, newAreas):
    
    # Make forward and reverse weights
    forward = make_barycentric_weights(cur_sphere, new_sphere)
    reverse = make_barycentric_weights(new_sphere, cur_sphere)
    
    n_new_nodes, n_old_nodes = len(forward), len(reverse)

    # Init reverse gather
    reverse_gather = [dict() for _ in range(n_new_nodes)]

    # Convert scattering weights to gathering weights
    for old_node in range(n_old_nodes):
        for key in reverse[old_node]:
            reverse_gather[key][old_node] = reverse[old_node][key]

    # Fill in adap gather
    adap_gather = [dict() for i in range(n_new_nodes)]
    for new_node in range(n_new_nodes):

        # Build set of all nodes used by forward weights
        forward_used = set(forward[new_node])

        # If key from reverse gather not in forward used set
        # the reverse scatter weights include something
        # the forward gather weights don't, so use reverse scatter    
        use_forward = True
        if len(set(reverse_gather[new_node]) - forward_used) > 0:
            use_forward = False

        if use_forward:
            adap_gather[new_node] = forward[new_node]
        else:
            adap_gather[new_node] = reverse_gather[new_node]

        # Begin the process of area correction by multiplying by gathering node areas
        for key in adap_gather[new_node]:
            adap_gather[new_node][key] *= newAreas[new_node]

    # Sum the scattering weights to prepare for first normalization
    corr_sum = np.zeros(n_old_nodes)
    for new_node in range(n_new_nodes):
        for key in adap_gather[new_node]:
             corr_sum[key] += adap_gather[new_node][key]

    # Normalize adap_gather so that each nodes weights
    # add up to 1.
    for new_node in range(n_new_nodes):

        # Divide by scatter scum and multiply by current area
        weightsum = np.float128(0)
        for key in adap_gather[new_node]:
            adap_gather[new_node][key] *= curAreas[key] / corr_sum[key]
            weightsum += adap_gather[new_node][key]

        # Normalize by weightsum
        if weightsum != 0:
            for key in adap_gather[new_node]:
                adap_gather[new_node][key] /= weightsum
                
    return adap_gather

def resample_surface_by_weights(input_surf, weights):
    
    # Init output re-sampled surf
    output_surf = np.zeros(len(weights))

    # Multiply value of input surf at each node by the weight
    # computed for this value in the new surface
    for i in range(len(weights)):

        accum = np.float128(0)
        for node in weights[i]:
            accum += input_surf[node] * weights[i][node]

        # The new value is then just the sum of the normalized weights
        output_surf[i] = accum
        
    return output_surf


def resample_surface(input_surf, cur_sphere, new_sphere, cur_area, new_area):

    # Load spheres
    cur_sphere = SurfaceFile(cur_sphere)
    new_sphere = SurfaceFile(new_sphere)

    # Change coords to matched radius
    cur_sphere.change_radius(100)
    new_sphere.change_radius(100)

    # Load areas as np array
    curAreas = nib.load(cur_area).agg_data()
    newAreas = nib.load(new_area).agg_data()

    # Generate weights with area adaption
    weights = gen_weights_adap_bary_area(cur_sphere, new_sphere, curAreas, newAreas)

    # Resample according to calculated weights
    new_surf = resample_surface_by_weights(input_surf, weights)

    return new_surf