import os
import nibabel as nib
from ...loading import load
from ...transform import add_surface_medial_walls, SurfLabels
from ..plot import plot
import numpy as np
import matplotlib.pyplot as plt

nib.imageglobals.logger.level = 40
file_dr = os.path.dirname(os.path.realpath(__file__))

def test_plotting_and_transforms_case():
    
    # Load ROI labels
    glasser_labels = np.random.randint(0, 10, size=64984)
    
    # Load without transforms make sure works
    data = np.random.random(size=64984)

    # Test generic smart plot function for
    # just making sure works w/o errors
    plot(data, colorbar=True)

    # Test Extract ROIs
    surf_labels = SurfLabels(glasser_labels, vectorize=False)
    rois = surf_labels.fit_transform(data)
    assert rois.shape[0] == 9

    # Test plot again, and also reverse transform function
    plot(surf_labels.inverse_transform(rois), colorbar=True)

    plt.close()