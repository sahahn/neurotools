from nibabel.gifti import gifti
import numpy as np
from nibabel.freesurfer.io import read_geometry
from nilearn.surface import load_surf_mesh

def geo_to_gifti(geo_coords, geo_faces):

    coords = gifti.GiftiDataArray(data=np.asarray(geo_coords, 'float32'),
                                  intent=gifti.intent_codes.field1['pointset'],
                                  datatype=16,
                                  endian=2,
                                  coordsys=gifti.GiftiCoordSystem(dataspace=3, xformspace=3))

    faces = gifti.GiftiDataArray(data=np.asarray(geo_faces, np.int32),
                                 intent=gifti.intent_codes.field1['triangle'],
                                 datatype=8,
                                 endian=2,
                                 coordsys=gifti.GiftiCoordSystem(dataspace=3, xformspace=3))

    img = gifti.GiftiImage()
    img.add_gifti_data_array(coords)
    img.add_gifti_data_array(faces)
    
    return img

def data_to_gifti(s_data):
    
    data = gifti.GiftiDataArray(data=np.asarray(s_data, 'float32'),
                                intent=11,
                                datatype=16,
                                endian=2)
    
    img = gifti.GiftiImage()
    img.add_gifti_data_array(data)
    
    return img

def load_geo_as_network(geo):
    '''Quick utility for loading a geometry file / just
    set of triangles as a networkx network.'''
    
    if isinstance(geo, 'str'):
        
        try:
            _, geo = load_surf_mesh(geo)
        except ValueError:
            _, geo = read_geometry(geo)
    
    # If passed as tuple of mesh
    elif len(geo) == 2:
        geo = geo[1]
        
    G = nx.Graph()
    
    for tri in geo:
        G.add_edge(tri[0], tri[1])
        G.add_edge(tri[0], tri[2])
        G.add_edge(tri[1], tri[2])
    
    return G
    
