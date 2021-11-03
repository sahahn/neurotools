import matplotlib.pyplot
import os
import nibabel as nib
from ...loading import load
from ...transform import add_surface_medial_walls, SurfLabels
from ..plot import plot

nib.imageglobals.logger.level = 40
file_dr = os.path.dirname(os.path.realpath(__file__))

def test_plotting_and_transforms_case():
    
    # Load ROI labels
    glasser_labels = load(os.path.join(file_dr, 'test_data', 'hcp_mmp.npy'))
    assert glasser_labels.shape == (64984,)
    
    # Load without transforms make sure works
    cifti_loc = os.path.join(file_dr, 'test_data', 'ex_32k_fs_LR.dscalar.nii')
    data = load(cifti_loc)
    assert data.shape == (59412,)
    
    # Load with transform func
    t_data = add_surface_medial_walls(cifti_loc)
    assert t_data.shape == (64984,)
    assert len(t_data[t_data!=0]) == len(data[data!=0])

    # Test generic smart plot function for
    # just making sure works w/o errors
    plot(t_data, colorbar=True)

    # Test Extract ROIs
    surf_labels = SurfLabels(glasser_labels, vectorize=False)
    rois = surf_labels.fit_transform(t_data)
    assert rois.shape == (360,)

    # Test plot again, and also reverse transform function
    plot(surf_labels.inverse_transform(rois), colorbar=True)