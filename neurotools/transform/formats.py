from nibabel.gifti import gifti
import numpy as np

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