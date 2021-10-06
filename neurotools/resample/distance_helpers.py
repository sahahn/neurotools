import numpy as np
from heapq import heappush, heappop
from .helper_funcs import flatten, get_coord_vec_length, normalize_vector


def get_signed_distance_helper(surf):

    sdh_base = SignedDistanceHelperBase(surf)
    sdh = SignedDistanceHelper(sdh_base)
    return sdh

class SignedDistanceHelperBase():
    
    NUM_TRIS_TO_TEST = 50
    NUM_TRIS_TEST_INCR = 50
    
    def __init__(self, surf):
        
        # Get min max coords
        self.min_coord = surf.coords.min(axis=0)
        self.max_coord = surf.coords.max(axis=0)
        
        # Init Oct
        self.root_oct = Oct(self.min_coord, self.max_coord)
        
        # Also keep track of not flatten for conv.
        self.coords = surf.coords.copy()
        
        # Keep track of un-flattened
        self.m_tris = surf.tris.copy()
        
        # Keep track of number of triangles
        self.m_numTris = len(self.m_tris)
        
        # For each triangle
        for tri_indx, thisTri in enumerate(self.m_tris):
            
            # Computing min_coord and max_coord
            # as 3D coords, just min across all three
            # vertex in triangle for each x,y,z and max
            tri_coords = self.coords[thisTri]
            
            min_coord = tri_coords.min(axis=0)
            max_coord = tri_coords.max(axis=0)
            
            # Add triangle -use bounding box for now as an easy
            # test to capture any chance of the triangle intersecting the Oct
            self.addTriangle(self.root_oct, tri_indx, min_coord, max_coord)
               
    def addTriangle(self, thisOct, tri_indx, min_coord, max_coord):
        
        # If leaf
        if thisOct.m_leaf:
            
            # Add triangle to list - where triangle is just
            # integer index of triangle
            thisOct.m_tri_indx_list.append(tri_indx)
            
            # Get current number of triangles in list
            numTris = len(thisOct.m_tri_indx_list)
            
            # Set by heuristics
            if (numTris >= self.NUM_TRIS_TO_TEST) and (
                (numTris % self.NUM_TRIS_TEST_INCR) == (self.NUM_TRIS_TO_TEST % self.NUM_TRIS_TEST_INCR)):
                
                totalSize = 0
                numSplit = 0
                
                # For each triangle
                for temp_tri_indx in thisOct.m_tri_indx_list:
                    
                    # Get triangle, then coords of triangle
                    tempTri = self.m_tris[temp_tri_indx]
                    tri_coords = self.coords[tempTri]
                    
                    # Get min and max
                    tempmin_coord = tri_coords.min(axis=0)
                    tempmax_coord = tri_coords.max(axis=0)
                    
                    # Basically sets to 0 for each point if
                    # less than midpoint or 1 if greater than
                    minOct, maxOct = np.zeros(3), np.zeros(3)
                    thisOct.containingChild(tempmin_coord, minOct)
                    thisOct.containingChild(tempmax_coord, maxOct)
                    
                    # If any are the same, that is to say
                    # the minimum of the point and the maximum of
                    # the point are both less than or both greater
                    # than the midpoint of the Oct, reduce the split size.
                    splitSize = 8
                    if (minOct[0] == maxOct[0]):
                        splitSize >>= 1
                    if (minOct[1] == maxOct[1]):
                        splitSize >>= 1
                    if (minOct[2] == maxOct[2]):
                        splitSize >>= 1
                        
                    totalSize += splitSize
                    
                    # So only in the case that one of the coordinates
                    # are both greater or less than the midpoint do
                    # we increment numSplit, not sure why... 
                    if (splitSize != 8):
                        numSplit += 1
                        
                # Don't split if all triangles end up in all child oct's
                # and try to balance speedup with memory usage.
                if numSplit > 0 and totalSize < (3 * numTris):
                    
                    # Do the split
                    thisOct.makeChildren()
                    
                    # For each triangle
                    for temp_tri_indx in thisOct.m_tri_indx_list:
                
                        # Get triangle, then coords of triangle
                        tempTri = self.m_tris[temp_tri_indx]
                        tri_coords = self.coords[tempTri]

                        # Get min and max
                        tempmin_coord = tri_coords.min(axis=0)
                        tempmax_coord = tri_coords.max(axis=0)
                        
                        # Iterate through each child
                        # Check if bounds overlap, if overlap add triangle
                        for child in flatten(thisOct.m_children):
                            if child.boundsOverlaps(tempmin_coord, tempmax_coord):
                                self.addTriangle(child, temp_tri_indx, tempmin_coord, tempmax_coord)
                                        
                                        
        # If not leaf - add tri to child Oct's only if overlaps
        else:

            for child in flatten(thisOct.m_children):
                if child.boundsOverlaps(min_coord, max_coord):
                    self.addTriangle(child, tri_indx, min_coord, max_coord)
                            
class SignedDistanceHelper():
    
    def __init__(self, myBase):
        
        self.base = myBase
        self.numTris = self.base.m_numTris
        self.m_triMarked = np.zeros(self.numTris)
        
    def barycentricWeights(self, coord):
        
        # Init heap, will store Oct's w/ associated float
        myHeap = []
        
        # Store the dist first, then index Oct, push to heap
        i_oct = self.base.root_oct
        heappush(myHeap, (i_oct.distToPoint(coord), i_oct))
        
        # Init starter vars
        bestInfo = None
        tempf, bestTriDist = -1, -1
        first = True
        
        # Until heap is empty
        while len(myHeap) > 0:
            
            tempf, curOct = heappop(myHeap)

            if first or tempf < bestTriDist:
                
                # Oct is leaf case
                if curOct.m_leaf:

                    # Iterate over each triangle indx in this oct
                    for tri_indx in curOct.m_tri_indx_list:
                        
                        # Only proceed if un-marked / unseen
                        if self.m_triMarked[tri_indx] != 1:
                            self.m_triMarked[tri_indx] = 1
                            
                            # Get unsignedDistToTri from coordinate to triangle
                            tempf, tempInfo = self.unsignedDistToTri(coord, tri_indx)
                            
                            # Check if new best
                            if first or tempf < bestTriDist:
                                bestInfo = tempInfo
                                bestTriDist = tempf
                                first = False     

                # Parent Oct case
                else:
                    
                    # Iterate over child Oct's
                    for child_oct in flatten(curOct.m_children):
                        
                        # Get distance from child to coord
                        # and add to heap if less than best seen
                        tempf = child_oct.distToPoint(coord)
                        
                        if first or tempf < bestTriDist:
                            heappush(myHeap, (tempf, child_oct))
        
        # Clean marked
        self.m_triMarked[:] = 0
        
        # Init bary weights
        baryWeights = np.empty((3))
        
        # Get tri nodes
        triNodes = self.base.m_tris[bestInfo.tri_indx]
        
        # Handle by case
        if bestInfo.p_type == 2 or bestInfo.p_type == 'TRIANGLE':
            
            verts = self.base.coords[triNodes]
            vp = verts - bestInfo.tempPoint

            weight1 = get_coord_vec_length(np.cross(vp[1], vp[2]))
            weight2 = get_coord_vec_length(np.cross(vp[0], vp[2]))
            weight3 = get_coord_vec_length(np.cross(vp[0], vp[1]))
            weight_sum = weight1 + weight2 + weight3
                
            # Set weights
            baryWeights[0] = weight1 / weight_sum
            baryWeights[1] = weight2 / weight_sum
            baryWeights[2] = weight3 / weight_sum

        elif bestInfo.p_type == 1 or bestInfo.p_type == 'EDGE':
            
            vert1 = self.base.coords[bestInfo.node1]
            vert2 = self.base.coords[bestInfo.node2]
            v21hat = vert2 - vert1
            
            v21hat, origLength =\
                normalize_vector(v21hat, return_norm=True)
            
            tempf = v21hat.dot(bestInfo.tempPoint - vert1)
            weight2 = tempf / origLength
            weight1 = 1 - weight2
            
            # Assign to correct spots
            for i in range(3):
                
                if triNodes[i] == bestInfo.node1:
                    baryWeights[i] = weight1
                elif triNodes[i] == bestInfo.node2:
                    baryWeights[i] = weight2
                else:
                    baryWeights[i] = 0
            
        elif bestInfo.p_type == 0 or bestInfo.p_type == 'NODE':
            
            # Put weight as 1 in correct spot, other weights=0
            # if just one node
            for i in range(3):
                if triNodes[i] == bestInfo.node1:
                    baryWeights[i] = 1
                else:
                    baryWeights[i] = 0

        else:
            raise RuntimeError(f'unknown p_type: {bestInfo.p_type}')
            
        # Make sure weights are not negative
        baryWeights[baryWeights<0] = 0
        
        # Just need nodes and baryWeights
        return triNodes, baryWeights
            
    def unsignedDistToTri(self, coord, tri_indx):
        
        # Get nodes of triangle
        triNodes = self.base.m_tris[tri_indx]
        
        # Inits
        point = np.array(coord)
        
        # Tracks whether it is closest to a node, an edge, or the face
        p_type = 0
        
        # Track nodes involved
        node1, node2 = -1, -1
        
        # Keep track of best point found
        bestPoint = np.empty(3)
        
        # Verts contains coordinates for each node of the triangle
        verts = self.base.coords[triNodes]
        
        v10 = verts[1] - verts[0]
        xhat = normalize_vector(v10)
        v20 = verts[2] - verts[0]
        
        # To get orthogonal basis vectors for projection
        yhat, sanity =\
            normalize_vector(v20 - xhat * xhat.dot(v20), return_norm=True)
        
        # If the triangle is (mostly) degenerate
        # find the closest point on its edges instead
        # of trying to project to the NaN plane
        if sanity == 0 or np.abs(xhat.dot(yhat)) > .01:
            
            first = True
            
            # Track best squared length from edge to original point
            bestLengthSqr = -1
            
            # Consecutive vertices, does 2,0 then 0,1 then 1,2
            j = 2
            for i in range(3):
                
                norm, length =\
                    normalize_vector(verts[j] - verts[i], return_norm=True)
                
                mypoint = np.empty((3))
                temptype = 0
                tempnode1, tempnode2 = -1, -1
                
                if length > 0:
                    diff = point - verts[i]
                    dot = norm.dot(diff)
                    
                    if dot <= 0:
                        mypoint = verts[i]
                        tempnode1 = triNodes[i]
                    elif dot >= length:
                        mypoint = verts[j]
                        tempnode1 = triNodes[j]
                    else:
                        mypoint = verts[i] + dot * norm
                        tempnode1 = triNodes[i]
                        tempnode2 = triNodes[j]
                        temptype = 1
        
                else:
                    temptype = 0
                    tempnode1 = triNodes[i]
                    mypoint = verts[i]
                    
                # Get distance squared
                td = (point - mypoint)
                tempdistsqr = np.sum(td*td)
                
                if first or tempdistsqr < bestLengthSqr:
                    first = False
                    p_type = temptype
                    bestLengthSqr = tempdistsqr
                    bestPoint = mypoint
                    node1 = tempnode1
                    node2 = tempnode2
                    
                # Consecutive vertices, does 2,0 then 0,1 then 1,2
                j = i
                
        else:
            
            # Project everything to the new plane with basis vectors xhat, yhat
            vertxy = np.empty((3, 2))
            for i in range(3):
                vertxy[i][0] = xhat.dot(verts[i] - verts[0])
                vertxy[i][1] = yhat.dot(verts[i] - verts[0])
                
            inside = True
            
            p = [xhat.dot(point - verts[0]),
                    yhat.dot(point - verts[0])]
            bestxy = np.empty((2))
            bestDist = -1
            
            # Go through cases
            j, k = 2, 1
            for i in range(3):
                
                norm = [vertxy[j][0] - vertxy[i][0],
                        vertxy[j][1] - vertxy[i][1]]
                diff = [p[0] - vertxy[i][0],
                        p[1] - vertxy[i][1]]
                direction = [vertxy[k][0] - vertxy[i][0],
                                vertxy[k][1] - vertxy[i][1]]
                edgelen = np.sqrt((norm[0] * norm[0]) + (norm[1] * norm[1]))
                
                # Non-Zero case
                if edgelen != 0:
                    
                    norm[0] /= edgelen
                    norm[1] /= edgelen
                    
                    dot = (direction[0] * norm[0]) + (direction[1] * norm[1])
                    
                    # Direction is orthogonal to norm, in the direction of the third vertex
                    direction[0] -= dot * norm[0]
                    direction[1] -= dot * norm[1]
                    
                    # If dot product with (projected point - vert[i]) is negative
                    # then we are outside the triangle, find the projection to this edge
                    # and break if it is the second time or otherwise known to be finished
                    if (diff[0] * direction[0]) + (diff[1] * direction[1]) < 0:
                        
                        inside = False
                        dot = (diff[0] * norm[0]) + (diff[1] * norm[1])
                        
                        if bestDist < 0:
                            
                            # If closest point on this edge is an endpoint,
                            # it is possible for another edge that we count as outside
                            # of to have a closer point
                            if dot <= 0:
                                p_type = 0
                                node1 = triNodes[i]
                                bestPoint = verts[i]
                                bestxy[0] = vertxy[i][0]
                                bestxy[1] = vertxy[i][1]
                            
                            elif dot >= edgelen:
                                p_type = 0
                                node1 = triNodes[j]
                                bestPoint = verts[j]
                                bestxy[0] = vertxy[j][0]
                                bestxy[1] = vertxy[j][1]
                            
                            # If closest point on the edge is in the middle of the edge,
                            # nothing can be closer, break
                            else:
                                p_type = 1
                                node1 = triNodes[i]
                                node2 = triNodes[j]
                                bestxy[0] = vertxy[i][0] + dot * norm[0]
                                bestxy[1] = vertxy[i][1] + dot * norm[1]
                                break
                                
                            diff[0] = p[0] - bestxy[0]
                            diff[1] = p[1] - bestxy[1]
                            bestDist = (diff[0] * diff[0]) + (diff[1] * diff[1])
                            
                        else:
                            tempnode1 = None
                            tempbestPoint = np.empty((3))
                            tempxy = np.empty((2))
                            
                            if dot <= 0:
                                tempnode1 = triNodes[i]
                                tempbestPoint = verts[i]
                                tempxy[0] = vertxy[i][0]
                                tempxy[1] = vertxy[i][1]

                            elif dot >= edgelen:
                                tempnode1 = triNodes[j]
                                tempbestPoint = verts[j]
                                tempxy[0] = vertxy[j][0]
                                tempxy[1] = vertxy[j][1]
                                
                            # Again, middle of edge always wins, don't bother with the extra test
                            else:
                                p_type = 1
                                node1 = triNodes[i]
                                node2 = triNodes[j]
                                bestxy[0] = vertxy[i][0] + dot * norm[0]
                                bestxy[1] = vertxy[i][1] + dot * norm[1]
                                break
                            
                            # Compute diffs
                            diff[0] = p[0] - tempxy[0]
                            diff[1] = p[1] - tempxy[1]
                            
                            tempdist = (diff[0] * diff[0]) + (diff[1] * diff[1])
                            
                            if tempdist < bestDist:
                                
                                # If it were in the middle of the edge, we wouldn't be here
                                p_type = 0
                                node1 = tempnode1
                                bestPoint = tempbestPoint
                                bestxy[0] = tempxy[0]
                                bestxy[1] = tempxy[1]
                            
                            # This is our second time outside an edge,
                            # we have now covered all 3 possible endpoints, so break
                            break
                            
                # Since we don't have an edge, we don't need to
                # othrogonalize direction, or project to the edge
                else:
                    
                    if (diff[0] * direction[0]) + (diff[1] * direction[1]) < 0:
                        inside = False
                        p_type = 0
                        node1 = triNodes[i]
                        bestPoint = verts[i]
                        break
                    
                # Consecutive vertices, does 2,0 then 0,1 then 1,2
                k = j
                j = i
                
            # Now outside of loop
            if inside:
                bestxy[0] = p[0]
                bestxy[1] = p[1]
                p_type = 2
            
            if p_type != 0:
                bestPoint = (bestxy[0] * xhat) + (bestxy[1] * yhat) + verts[0]
            
        # result is then just point - bestPoint
        result = point - bestPoint

        myInfo = ClosestPointInfo(p_type=p_type,
                                    node1=node1,
                                    node2=node2,
                                    tri_indx=tri_indx,
                                    tempPoint=bestPoint)

        return get_coord_vec_length(result), myInfo

class Oct():
    
    def __init__(self, min_coords=None, max_coords=None):
        
        # Save args
        self.min_coords = min_coords
        self.max_coords = max_coords
        
        # Init children pointers
        self.m_children = [[[None for k in range(2)] for j in range(2)]
                           for i in range(2)]

        # Parent pointer
        self.m_parent = np.NaN
        
        # Boolean indicate if leaf
        self.m_leaf = True
        
        # Stores list of triangle index's
        # in reference to index of triangle in master
        # list
        self.m_tri_indx_list = []
        
        # Init
        self.m_bounds = np.zeros((3, 3))
        
        # Setup bounds
        if min_coords is not None and max_coords is not None:
            self.m_bounds[:,0] = self.min_coords
            self.m_bounds[:,1] = (self.min_coords + self.max_coords) / 2
            self.m_bounds[:,2] = self.max_coords
        
    def containingChild(self, point, whichOct=None):
        
        # Init
        myOct = np.zeros((3)).astype('int')
                
        for i in range(3):
            
            # If strictly less than, using only the midpoint is how traversal works
            # even if the point isn't inside the Oct
            if point[i] < self.m_bounds[i][1]:
                myOct[i] = 0
            else:
                myOct[i] = 1
            
            # Modify passed array if passed
            if whichOct is not None:
                whichOct[i] = myOct[i]

        return self.m_children[myOct[0]][myOct[1]][myOct[2]]
    
    def makeChildren(self):
        
        self.m_leaf = False
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    
                    # Gen new Oct
                    temp = Oct()
                    
                    # With parent as this Oct
                    temp.m_parent = self
                    
                    # Fill in new m_bounds
                    ijk = [i,j,k]
                    for m in range(3):
                        temp.m_bounds[m][0] = self.m_bounds[m][ijk[m]]
                        temp.m_bounds[m][2] = self.m_bounds[m][ijk[m] + 1]
                        temp.m_bounds[m][1] = (temp.m_bounds[m][0] + temp.m_bounds[m][2]) * 0.5

                    # Add to children array
                    self.m_children[i][j][k] = temp
                    
    def boundsOverlaps(self, min_coords, max_coords):
        
        for i in range(3):
            if (max_coords[i] < self.m_bounds[i][0]) or (min_coords[i] > self.m_bounds[i][2]):
                return False

        return True
    
    def distToPoint(self, point):
        
        temp = np.empty(3)
        
        for i in range(3):
            
            if point[i] < self.m_bounds[i][0]:
                temp[i] = self.m_bounds[i][0] - point[i]
            else:
                if point[i] > self.m_bounds[i][2]:
                    temp[i] = self.m_bounds[i][2] - point[i]
                else:
                    temp[i] = 0

        return get_coord_vec_length(temp)
    
    def __repr__(self):
        
        if len(self.m_tri_indx_list) == 0:
            return ''
        
        if self.m_leaf:
            return f'Oct(n_tris={len(self.m_tri_indx_list)})'
        
        else:
    
            out = ''
            for child in flatten(self.m_children):
                r = repr(child)
                if len(r) > 1:
                    out += r + '\n'

        return out

class ClosestPointInfo():
    '''Struc used by SignedDistanceHelper in passing around info'''
    
    def __init__(self, p_type, node1, node2, tri_indx, tempPoint):
        
        self.p_type = p_type
        self.node1 = node1
        self.node2 = node2
        self.tri_indx = tri_indx
        self.tempPoint = tempPoint
        
    def __repr__(self):
        
        return f'p_type={self.p_type} node1={self.node1} node2={self.node2} tri_indx={self.tri_indx} tempPoint={self.tempPoint}'
