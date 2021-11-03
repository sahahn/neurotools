import numpy as np
import nibabel as nib
from .helper_funcs import triangle_area, get_coord_vec_length


class SurfaceFile():
    
    def __init__(self, loc):
        
        # Load as nifti
        # @TODO handle other formats?
        self.raw = nib.load(loc)
        
        # Get coords and tris
        self.coords = self.raw.agg_data('NIFTI_INTENT_POINTSET')
        self.tris = self.raw.agg_data('NIFTI_INTENT_TRIANGLE')

    def change_radius(self, radius=100):
        '''Sets modified coords to matched radius'''
        
        # Check if already correct radius
        if all(self.coords.max(axis=0) == radius) and all(self.coords.min(axis=0) == -radius):
            return
        
        # Otherwise change
        new_coords = np.zeros((self.coords.shape))
        for i, coord in enumerate(self.coords):
            new_coords[i] = coord * (radius / get_coord_vec_length(coord))

        self.coords = new_coords
        return
    
    def compute_vertex_areas(self):

        # Init areas
        areas = np.zeros(np.unique(self.tris).shape)

        for tri in self.tris:
            
            # Unpack
            n1, n2, n3 = tri
            
            # Compute area / 3
            area3 = triangle_area(self.coords[n1], self.coords[n2], self.coords[n3]) / 3
            
            # Add to areas out
            areas[n1] += area3
            areas[n2] += area3
            areas[n3] += area3

        return areas
