import numpy as np
from heapq import heappush, heappop
from .helper_funcs import flatten, get_coord_vec_length, normalize_vector
import time


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
        self.tris = surf.tris.copy()
        self.n_tris = len(self.tris)

        start = time.time()
        
        # For each triangle
        for tri_indx, tri in enumerate(self.tris):
            
            # Computing min_coord and max_coord
            # as 3D coords, just min across all three
            # vertex in triangle for each x,y,z and max
            tri_coords = self.coords[tri]
            
            min_coord = tri_coords.min(axis=0)
            max_coord = tri_coords.max(axis=0)
            
            # Add triangle -use bounding box for now as an easy
            # test to capture any chance of the triangle intersecting the Oct
            self.add_triangle(self.root_oct, tri_indx, min_coord, max_coord)

        print(f'Added triangles to signed helper base in {time.time() - start}')
               
    def add_triangle(self, c_oct, tri_indx, min_coord, max_coord):
        
        # If leaf
        if c_oct.m_leaf:
            
            # Add triangle to list - where triangle is just
            # integer index of triangle
            c_oct.m_tri_indx_list.append(tri_indx)
            
            # Get current number of triangles in list
            n_tris = len(c_oct.m_tri_indx_list)
            
            # Set by heuristics
            if (n_tris >= self.NUM_TRIS_TO_TEST) and (
                (n_tris % self.NUM_TRIS_TEST_INCR) == (self.NUM_TRIS_TO_TEST % self.NUM_TRIS_TEST_INCR)):
                
                total_size = 0
                n_split = 0
                
                # For each triangle
                for temp_tri_indx in c_oct.m_tri_indx_list:
                    
                    # Get triangle, then coords of triangle
                    temp_tri = self.tris[temp_tri_indx]
                    tri_coords = self.coords[temp_tri]
                    
                    # Get min and max
                    tempmin_coord = tri_coords.min(axis=0)
                    tempmax_coord = tri_coords.max(axis=0)
                    
                    # Basically sets to 0 for each point if
                    # less than midpoint or 1 if greater than
                    min_oct, max_oct = np.zeros(3), np.zeros(3)
                    c_oct.containing_child(tempmin_coord, min_oct)
                    c_oct.containing_child(tempmax_coord, max_oct)
                    
                    # If any are the same, that is to say
                    # the minimum of the point and the maximum of
                    # the point are both less than or both greater
                    # than the midpoint of the Oct, reduce the split size.
                    split_sz = 8
                    if (min_oct[0] == max_oct[0]):
                        split_sz >>= 1
                    if (min_oct[1] == max_oct[1]):
                        split_sz >>= 1
                    if (min_oct[2] == max_oct[2]):
                        split_sz >>= 1
                        
                    total_size += split_sz
                    
                    # So only in the case that one of the coordinates
                    # are both greater or less than the midpoint do
                    # we increment n_split, not sure why... 
                    if (split_sz != 8):
                        n_split += 1
                        
                # Don't split if all triangles end up in all child oct's
                # and try to balance speedup with memory usage.
                if n_split > 0 and total_size < (3 * n_tris):
                    
                    # Do the split
                    c_oct.make_children()
                    
                    # For each triangle
                    for temp_tri_indx in c_oct.m_tri_indx_list:
                
                        # Get triangle, then coords of triangle
                        temp_tri = self.tris[temp_tri_indx]
                        tri_coords = self.coords[temp_tri]

                        # Get min and max
                        tempmin_coord = tri_coords.min(axis=0)
                        tempmax_coord = tri_coords.max(axis=0)
                        
                        # Iterate through each child
                        # Check if bounds overlap, if overlap add triangle
                        for child in flatten(c_oct.m_children):
                            if child.bounds_overlaps(tempmin_coord, tempmax_coord):
                                self.add_triangle(child, temp_tri_indx, tempmin_coord, tempmax_coord)
                                        
                                        
        # If not leaf - add tri to child Oct's only if overlaps
        else:

            for child in flatten(c_oct.m_children):
                if child.bounds_overlaps(min_coord, max_coord):
                    self.add_triangle(child, tri_indx, min_coord, max_coord)
                            
class SignedDistanceHelper():
    
    def __init__(self, base):
        
        self.base = base
        self.tri_markers = np.zeros(self.base.n_tris)
        
    def barycentric_weights(self, coord):
        
        # Init heap, will store Oct's w/ associated float
        heap = []
        
        # Store the dist first, then index Oct, push to heap
        r_oct = self.base.root_oct
        heappush(heap, (r_oct.dist_to_point(coord), r_oct))
        
        # Init starter vars
        tempf, best_tri_dist = -1, -1
        best_info, first = None, True
        
        # Until heap is empty
        while len(heap) > 0:
            
            tempf, curOct = heappop(heap)

            if first or tempf < best_tri_dist:
                
                # Oct is leaf case
                if curOct.m_leaf:

                    # Iterate over each triangle indx in this oct
                    for tri_indx in curOct.m_tri_indx_list:
                        
                        # Only proceed if un-marked / unseen
                        if self.tri_markers[tri_indx] != 1:
                            self.tri_markers[tri_indx] = 1
                            
                            # Get unsigned_dist_to_tri from coordinate to triangle
                            tempf, temp_info = self.unsigned_dist_to_tri(coord, tri_indx)
                            
                            # Check if new best
                            if first or tempf < best_tri_dist:
                                best_info = temp_info
                                best_tri_dist = tempf
                                first = False     

                # Parent Oct case
                else:
                    
                    # Iterate over child Oct's
                    for child_oct in flatten(curOct.m_children):
                        
                        # Get distance from child to coord
                        # and add to heap if less than best seen
                        tempf = child_oct.dist_to_point(coord)
                        
                        if first or tempf < best_tri_dist:
                            heappush(heap, (tempf, child_oct))
        
        # Clean marked
        self.tri_markers[:] = 0
        
        # Init bary weights
        bary_weights = np.empty((3))
        
        # Get tri nodes
        tri_nodes = self.base.tris[best_info.tri_indx]
        
        # Handle by case
        if best_info.p_type == 2 or best_info.p_type == 'TRIANGLE':
            
            verts = self.base.coords[tri_nodes]
            vp = verts - best_info.temp_point

            weight1 = get_coord_vec_length(np.cross(vp[1], vp[2]))
            weight2 = get_coord_vec_length(np.cross(vp[0], vp[2]))
            weight3 = get_coord_vec_length(np.cross(vp[0], vp[1]))
            weight_sum = weight1 + weight2 + weight3
                
            # Set weights
            bary_weights[0] = weight1 / weight_sum
            bary_weights[1] = weight2 / weight_sum
            bary_weights[2] = weight3 / weight_sum

        elif best_info.p_type == 1 or best_info.p_type == 'EDGE':
            
            vert1 = self.base.coords[best_info.node1]
            vert2 = self.base.coords[best_info.node2]
            v21hat = vert2 - vert1
            
            v21hat, orig_len =\
                normalize_vector(v21hat, return_norm=True)
            
            tempf = v21hat.dot(best_info.temp_point - vert1)
            weight2 = tempf / orig_len
            weight1 = 1 - weight2
            
            # Assign to correct spots
            for i in range(3):
                
                if tri_nodes[i] == best_info.node1:
                    bary_weights[i] = weight1
                elif tri_nodes[i] == best_info.node2:
                    bary_weights[i] = weight2
                else:
                    bary_weights[i] = 0
            
        elif best_info.p_type == 0 or best_info.p_type == 'NODE':
            
            # Put weight as 1 in correct spot, other weights=0
            # if just one node
            for i in range(3):
                if tri_nodes[i] == best_info.node1:
                    bary_weights[i] = 1
                else:
                    bary_weights[i] = 0

        else:
            raise RuntimeError(f'unknown p_type: {best_info.p_type}')
            
        # Make sure weights are not negative
        bary_weights[bary_weights<0] = 0
        
        # Just need nodes and bary_weights
        return tri_nodes, bary_weights
            
    def unsigned_dist_to_tri(self, coord, tri_indx):
        
        # Get nodes of triangle
        tri_nodes = self.base.tris[tri_indx]
        
        # Inits
        point = np.array(coord)
        
        # Tracks whether it is closest to a node, an edge, or the face
        p_type = 0
        
        # Track nodes involved
        node1, node2 = -1, -1
        
        # Keep track of best point found
        best_point = np.empty(3)
        
        # Verts contains coordinates for each node of the triangle
        verts = self.base.coords[tri_nodes]
        
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
            best_len_sqr = -1
            
            # Consecutive vertices, does 2,0 then 0,1 then 1,2
            j = 2
            for i in range(3):
                
                norm, length =\
                    normalize_vector(verts[j] - verts[i], return_norm=True)
                
                mypoint = np.empty((3))
                temptype = 0
                temp_node1, tempnode2 = -1, -1
                
                if length > 0:
                    diff = point - verts[i]
                    dot = norm.dot(diff)
                    
                    if dot <= 0:
                        mypoint = verts[i]
                        temp_node1 = tri_nodes[i]
                    elif dot >= length:
                        mypoint = verts[j]
                        temp_node1 = tri_nodes[j]
                    else:
                        mypoint = verts[i] + dot * norm
                        temp_node1 = tri_nodes[i]
                        tempnode2 = tri_nodes[j]
                        temptype = 1
        
                else:
                    temptype = 0
                    temp_node1 = tri_nodes[i]
                    mypoint = verts[i]
                    
                # Get distance squared
                td = (point - mypoint)
                temp_dist_sqr = np.sum(td*td)
                
                if first or temp_dist_sqr < best_len_sqr:
                    first = False
                    p_type = temptype
                    best_len_sqr = temp_dist_sqr
                    best_point = mypoint
                    node1 = temp_node1
                    node2 = tempnode2
                    
                # Consecutive vertices, does 2,0 then 0,1 then 1,2
                j = i
                
        else:
            
            # Project everything to the new plane with basis vectors xhat, yhat
            vert_xy = np.empty((3, 2))
            for i in range(3):
                vert_xy[i][0] = xhat.dot(verts[i] - verts[0])
                vert_xy[i][1] = yhat.dot(verts[i] - verts[0])
                
            inside = True
            
            p = [xhat.dot(point - verts[0]),
                 yhat.dot(point - verts[0])]
            best_xy = np.empty((2))
            best_dist = -1
            
            # Go through cases
            j, k = 2, 1
            for i in range(3):
                
                norm = [vert_xy[j][0] - vert_xy[i][0],
                        vert_xy[j][1] - vert_xy[i][1]]
                diff = [p[0] - vert_xy[i][0],
                        p[1] - vert_xy[i][1]]
                direction = [vert_xy[k][0] - vert_xy[i][0],
                             vert_xy[k][1] - vert_xy[i][1]]
                edge_len = np.sqrt((norm[0] * norm[0]) + (norm[1] * norm[1]))
                
                # Non-Zero case
                if edge_len != 0:
                    
                    norm[0] /= edge_len
                    norm[1] /= edge_len
                    
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
                        
                        if best_dist < 0:
                            
                            # If closest point on this edge is an endpoint,
                            # it is possible for another edge that we count as outside
                            # of to have a closer point
                            if dot <= 0:
                                node1, best_point = tri_nodes[i], verts[i]
                                best_xy[0], best_xy[1] = vert_xy[i][0], vert_xy[i][1]
                            
                            elif dot >= edge_len:
                                node1, best_point = tri_nodes[j], verts[j]
                                best_xy[0], best_xy[1] = vert_xy[j][0], vert_xy[j][1]
                            
                            # If closest point on the edge is in the middle of the edge,
                            # nothing can be closer, break
                            else:
                                p_type = 1
                                node1, node2 = tri_nodes[i], tri_nodes[j]
                                best_xy[0] = vert_xy[i][0] + dot * norm[0]
                                best_xy[1] = vert_xy[i][1] + dot * norm[1]
                                break
                                
                            diff[0], diff[1] = p[0] - best_xy[0], p[1] - best_xy[1]
                            best_dist = (diff[0] * diff[0]) + (diff[1] * diff[1])
                            
                        else:

                            temp_node1 = None
                            temp_best_point = np.empty((3))
                            temp_xy = np.empty((2))
                            
                            if dot <= 0:
                                temp_node1, temp_best_point = tri_nodes[i], verts[i]
                                temp_xy[0], temp_xy[1] = vert_xy[i][0], vert_xy[i][1]

                            elif dot >= edge_len:
                                temp_node1 = tri_nodes[j]
                                temp_best_point = verts[j]
                                temp_xy[0], temp_xy[1] = vert_xy[j][0], vert_xy[j][1]
                                
                            # Again, middle of edge always wins, don't bother with the extra test
                            else:
                                p_type = 1
                                node1, node2 = tri_nodes[i], tri_nodes[j]
                                best_xy[0] = vert_xy[i][0] + dot * norm[0]
                                best_xy[1] = vert_xy[i][1] + dot * norm[1]
                                break
                            
                            # Compute diffs
                            diff[0] = p[0] - temp_xy[0]
                            diff[1] = p[1] - temp_xy[1]
                            
                            temp_dist = (diff[0] * diff[0]) + (diff[1] * diff[1])
                            
                            if temp_dist < best_dist:
                                
                                # If it were in the middle of the edge, we wouldn't be here
                                p_type = 0
                                node1, best_point = temp_node1, temp_best_point
                                best_xy[0], best_xy[1] = temp_xy[0], temp_xy[1]
                            
                            # This is our second time outside an edge,
                            # we have now covered all 3 possible endpoints, so break
                            break
                            
                # Since we don't have an edge, we don't need to
                # othrogonalize direction, or project to the edge
                else:
                    
                    if (diff[0] * direction[0]) + (diff[1] * direction[1]) < 0:
                        inside = False
                        p_type = 0
                        node1 = tri_nodes[i]
                        best_point = verts[i]
                        break
                    
                # Consecutive vertices, does 2,0 then 0,1 then 1,2
                k = j
                j = i
                
            # Now outside of loop
            if inside:
                best_xy[0] = p[0]
                best_xy[1] = p[1]
                p_type = 2
            
            if p_type != 0:
                best_point = (best_xy[0] * xhat) + (best_xy[1] * yhat) + verts[0]
            
        # result is then just point - best_point
        result = point - best_point

        myInfo = ClosestPointInfo(p_type=p_type,
                                  node1=node1,
                                  node2=node2,
                                  tri_indx=tri_indx,
                                  temp_point=best_point)

        return get_coord_vec_length(result), myInfo

class Oct():
    
    def __init__(self, min_coords=None, max_coords=None):
        
        # Save args
        self.min_coords = min_coords
        self.max_coords = max_coords
        
        # Init children pointers
        self.m_children = [[[None for _ in range(2)] for _ in range(2)] for _ in range(2)]

        # Parent pointer
        self.m_parent = np.nan
        
        # Boolean indicate if leaf
        self.m_leaf = True
        
        # Stores list of triangle index's in
        # reference to index of triangle in master list
        self.m_tri_indx_list = []
        
        # Init
        self.m_bounds = np.zeros((3, 3))
        
        # Setup bounds
        if min_coords is not None and max_coords is not None:
            self.m_bounds[:,0] = self.min_coords
            self.m_bounds[:,1] = (self.min_coords + self.max_coords) / 2
            self.m_bounds[:,2] = self.max_coords
        
    def containing_child(self, point, w_oct=None):
        
        # Init
        m_oct = np.zeros((3)).astype('int')
                
        for i in range(3):
            
            # If strictly less than, using only the midpoint is how traversal works
            # even if the point isn't inside the Oct
            if point[i] < self.m_bounds[i][1]:
                m_oct[i] = 0
            else:
                m_oct[i] = 1
            
            # Modify passed array if passed
            if w_oct is not None:
                w_oct[i] = m_oct[i]

        return self.m_children[m_oct[0]][m_oct[1]][m_oct[2]]
    
    def make_children(self):
        
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
                    
    def bounds_overlaps(self, min_coords, max_coords):
        
        for i in range(3):
            if (max_coords[i] < self.m_bounds[i][0]) or (min_coords[i] > self.m_bounds[i][2]):
                return False

        return True
    
    def dist_to_point(self, point):
        
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
    
    def __init__(self, p_type, node1, node2, tri_indx, temp_point):
        
        self.p_type = p_type
        self.node1 = node1
        self.node2 = node2
        self.tri_indx = tri_indx
        self.temp_point = temp_point
        
    def __repr__(self):
        
        return f'p_type={self.p_type} node1={self.node1} node2={self.node2} tri_indx={self.tri_indx} temp_point={self.temp_point}'
