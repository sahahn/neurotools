import numpy as np
import nibabel as nib
from .distance_helpers import get_signed_distance_helper
from .surface_file import SurfaceFile

def makeBarycentricWeights(from_s, to_s):
    
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

def gen_weights_adap_bary_area(curSphere, newSphere, curAreas, newAreas):
    
    # Make forward and reverse weights
    forward = makeBarycentricWeights(curSphere, newSphere)
    reverse = makeBarycentricWeights(newSphere, curSphere)
    
    numNewNodes, numOldNodes = len(forward), len(reverse)

    # Init reverse gather
    reverse_gather = [dict() for i in range(numNewNodes)]

    # Convert scattering weights to gathering weights
    for oldNode in range(numOldNodes):
        for key in reverse[oldNode]:
            reverse_gather[key][oldNode] = reverse[oldNode][key]

    # Fill in adap gather
    adap_gather = [dict() for i in range(numNewNodes)]
    for newNode in range(numNewNodes):

        # Build set of all nodes used by forward weights
        forwardused = set(forward[newNode])

        # If key from reverse gather not in forward used set
        # the reverse scatter weights include something
        # the forward gather weights don't, so use reverse scatter    
        useforward = True
        if len(set(reverse_gather[newNode]) - forwardused) > 0:
            useforward = False

        if useforward:
            adap_gather[newNode] = forward[newNode]
        else:
            adap_gather[newNode] = reverse_gather[newNode]

        # Begin the process of area correction by multiplying by gathering node areas
        for key in adap_gather[newNode]:
            adap_gather[newNode][key] *= newAreas[newNode]

    # Sum the scattering weights to prepare for first normalization
    correctionSum = np.zeros(numOldNodes)
    for newNode in range(numNewNodes):
        for key in adap_gather[newNode]:
             correctionSum[key] += adap_gather[newNode][key]

    # Normalize adap_gather so that each nodes weights
    # add up to 1.
    for newNode in range(numNewNodes):

        # Divide by scatter scum and multiply by current area
        weightsum = np.float128(0)
        for key in adap_gather[newNode]:
            adap_gather[newNode][key] *= curAreas[key] / correctionSum[key]
            weightsum += adap_gather[newNode][key]

        # Normalize by weightsum
        if weightsum != 0:
            for key in adap_gather[newNode]:
                adap_gather[newNode][key] /= weightsum
                
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


def resample_surface():

    # Define args
    input_surf = np.random.random((163842,))

    cur_sphere = 'fsaverage_std_sphere.L.164k_fsavg_L.surf.gii'
    new_sphere = 'fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii'

    cur_area = 'fsaverage.L.midthickness_va_avg.164k_fsavg_L.shape.gii'
    new_area = 'fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii'

    # Load spheres
    curSphere = SurfaceFile(cur_sphere)
    newSphere = SurfaceFile(new_sphere)

   # Change coords to matched radius
    curSphere.change_radius(100)
    newSphere.change_radius(100)

    # Load areas as np array
    curAreas = nib.load(cur_area).agg_data()
    newAreas = nib.load(new_area).agg_data()

    # Generate weights with area adaption
    weights = gen_weights_adap_bary_area(curSphere, newSphere, curAreas, newAreas)

    # Resample according to calculated weights
    new_surf = resample_surface_by_weights(input_surf, weights)

    return new_surf