import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colorbar import make_axes
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.colors import Normalize, LinearSegmentedColormap
from distutils.version import LooseVersion
import matplotlib

import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from nilearn.surface import load_surf_data, load_surf_mesh

def _crop_colorbar(cbar, cbar_vmin, cbar_vmax):
    """Crop a colorbar to show from cbar_vmin to cbar_vmax.
    Used when symmetric_cbar=False is used.
    """
    if (cbar_vmin is None) and (cbar_vmax is None):
        return
    cbar_tick_locs = cbar.locator.locs
    if cbar_vmax is None:
        cbar_vmax = cbar_tick_locs.max()
    if cbar_vmin is None:
        cbar_vmin = cbar_tick_locs.min()
    new_tick_locs = np.linspace(cbar_vmin, cbar_vmax,
                                len(cbar_tick_locs))

    # matplotlib >= 3.2.0 no longer normalizes axes between 0 and 1
    # See https://matplotlib.org/3.2.1/api/prev_api_changes/api_changes_3.2.0.html
    # _outline was removed in
    # https://github.com/matplotlib/matplotlib/commit/03a542e875eba091a027046d5ec652daa8be6863
    # so we use the code from there
    if LooseVersion(matplotlib.__version__) >= LooseVersion("3.2.0"):
        cbar.ax.set_ylim(cbar_vmin, cbar_vmax)
        X = cbar._mesh()[0]
        X = np.array([X[0], X[-1]])
        Y = np.array([[cbar_vmin, cbar_vmin], [cbar_vmax, cbar_vmax]])
        N = X.shape[0]
        ii = [0, 1, N - 2, N - 1, 2 * N - 1, 2 * N - 2, N + 1, N, 0]
        x = X.T.reshape(-1)[ii]
        y = Y.T.reshape(-1)[ii]
        xy = (np.column_stack([y, x])
              if cbar.orientation == 'horizontal' else
              np.column_stack([x, y]))
        cbar.outline.set_xy(xy)
    else:
        cbar.ax.set_ylim(cbar.norm(cbar_vmin), cbar.norm(cbar_vmax))
        outline = cbar.outline.get_xy()
        outline[:2, 1] += cbar.norm(cbar_vmin)
        outline[2:6, 1] -= (1. - cbar.norm(cbar_vmax))
        outline[6:, 1] += cbar.norm(cbar_vmin)
        cbar.outline.set_xy(outline)

    cbar.set_ticks(new_tick_locs)
    cbar.update_ticks()

VALID_VIEWS = "anterior", "posterior", "medial", "lateral", "dorsal", "ventral"
VALID_HEMISPHERES = "left", "right"

MATPLOTLIB_VIEWS = {"right": {"lateral": (0, 0),
                              "medial": (0, 180),
                              "dorsal": (90, 0),
                              "ventral": (270, 0),
                              "anterior": (0, 90),
                              "posterior": (0, 270)
                              },
                    "left": {"medial": (0, 0),
                             "lateral": (0, 180),
                             "dorsal": (90, 0),
                             "ventral": (270, 0),
                             "anterior": (0, 90),
                             "posterior": (0, 270)
                             }
                    }

def _set_view_plot_surf_matplotlib(hemi, view):
    """Helper function for plot_surf with matplotlib engine.
    This function checks the selected hemisphere and view, and
    returns elev and azim.
    """
    if hemi not in VALID_HEMISPHERES:
        raise ValueError(f"hemi must be one of {VALID_HEMISPHERES}")
    if view not in MATPLOTLIB_VIEWS[hemi]:
        raise ValueError(f"view must be one of {VALID_VIEWS}")
    return MATPLOTLIB_VIEWS[hemi][view]




def plot_single_surf(surf_mesh, surf_map=None, bg_map=None,
                     hemi='left', view='lateral', cmap=None, colorbar=False,
                     avg_method='mean', threshold=None, alpha='auto',
                     bg_on_data=False, darkness=1, vmin=None, vmax=None,
                     cbar_vmin=None, cbar_vmax=None,
                     title=None, output_file=None, axes=None, figure=None,
                     dist=8, cbar_tick_format='%.2g', **kwargs):

    _default_figsize = [4, 4]

    # load mesh and derive axes limits
    mesh = load_surf_mesh(surf_mesh)
    coords, faces = mesh[0], mesh[1]
    limits = [coords.min(), coords.max()]

    # Get elev and azim from view
    elev, azim = _set_view_plot_surf_matplotlib(hemi, view)

    # set alpha if in auto mode
    if alpha == 'auto':
        if bg_map is None:
            alpha = .5
        else:
            alpha = 1

    # if no cmap is given, set to matplotlib default
    if cmap is None:
        cmap = plt.cm.get_cmap(plt.rcParamsDefault['image.cmap'])
    else:
        # if cmap is given as string, translate to matplotlib cmap
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)

    figsize = _default_figsize
    
    # Leave space for colorbar
    if colorbar:
        figsize[0] += .7
    
    # Initiate figure and 3d axes
    if axes is None:
        if figure is None:
            figure = plt.figure(figsize=figsize)
        axes = figure.add_axes((0, 0, 1, 1), projection="3d")
    else:
        if figure is None:
            figure = axes.get_figure()

    axes.set_xlim(*limits)
    axes.set_ylim(*limits)
    axes.view_init(elev=elev, azim=azim)
    axes.set_axis_off()

    # plot mesh without data
    p3dcollec = axes.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                                  triangles=faces, linewidth=0.,
                                  antialiased=False,
                                  color='white')

    # reduce viewing distance to remove space around mesh
    axes.dist = dist

    # set_facecolors function of Poly3DCollection is used as passing the
    # facecolors argument to plot_trisurf does not seem to work
    face_colors = np.ones((faces.shape[0], 4))

    if bg_map is None:
        bg_data = np.ones(coords.shape[0]) * 0.5

    else:
        bg_data = load_surf_data(bg_map)
        if bg_data.shape[0] != coords.shape[0]:
            raise ValueError('The bg_map does not have the same number '
                             'of vertices as the mesh.')

    bg_faces = np.mean(bg_data[faces], axis=1)
    if bg_faces.min() != bg_faces.max():
        bg_faces = bg_faces - bg_faces.min()
        bg_faces = bg_faces / bg_faces.max()
    
    # Control background darkness
    bg_faces *= darkness
    face_colors = plt.cm.gray_r(bg_faces)

    # modify alpha values of background
    face_colors[:, 3] = alpha * face_colors[:, 3]
    # should it be possible to modify alpha of surf data as well?

    if surf_map is not None:
        surf_map_data = load_surf_data(surf_map)
        
        if len(surf_map_data.shape) != 1:
            raise ValueError('surf_map can only have one dimension but has'
                             '%i dimensions' % len(surf_map_data.shape))
        if surf_map_data.shape[0] != coords.shape[0]:
            raise ValueError('The surf_map does not have the same number '
                             'of vertices as the mesh.')

        # create face values from vertex values by selected avg methods
        if avg_method == 'mean':
            surf_map_faces = np.mean(surf_map_data[faces], axis=1)
        elif avg_method == 'median':
            surf_map_faces = np.median(surf_map_data[faces], axis=1)
        elif avg_method == 'min':
            surf_map_faces = np.min(surf_map_data[faces], axis=1)
        elif avg_method == 'max':
            surf_map_faces = np.max(surf_map_data[faces], axis=1)
        elif callable(avg_method):
            surf_map_faces =\
                np.apply_along_axis(avg_method, 1, surf_map_data[faces])

            ## check that surf_map_faces has the same length as face_colors
            if surf_map_faces.shape != (face_colors.shape[0],):
                raise ValueError(
                    'Array computed with the custom function '
                    'from avg_method does not have the correct shape: '
                    '{} != {}'.format(
                        surf_map_faces.shape[0],
                        face_colors.shape[0]
                    )
                )

            ## check that dtype is either int or float
            if not (
                "int" in str(surf_map_faces.dtype) or
                "float" in str(surf_map_faces.dtype)
            ):
                raise ValueError(
                    'Array computed with the custom function '
                    'from avg_method should be an array of numbers '
                    '(int or float)'
                )
        else:
            raise ValueError(
                "avg_method should be either "
                "['mean', 'median', 'max', 'min'] "
                "or a custom function"
            )

        # if no vmin/vmax are passed figure them out from data
        if vmin is None:
            vmin = np.nanmin(surf_map_faces)
        if vmax is None:
            vmax = np.nanmax(surf_map_faces)

        # Threshold if indicated
        if threshold is None:

            # If no thresholding and nans, filter them out
            kept_indices = np.where(
                            np.logical_not(
                                np.isnan(surf_map_faces)))[0]
        else:
            kept_indices = np.where(np.abs(surf_map_faces) >= threshold)[0]

        surf_map_faces = surf_map_faces - vmin
        surf_map_faces = surf_map_faces / (vmax - vmin)

        # multiply data with background if indicated
        if isinstance(bg_on_data, bool):
            if bg_on_data:
                face_colors[kept_indices] = cmap(surf_map_faces[kept_indices])\
                    * face_colors[kept_indices]
            else:
                face_colors[kept_indices] = cmap(surf_map_faces[kept_indices])
        else:
            face_colors[kept_indices] = cmap(surf_map_faces[kept_indices])\
                    * plt.cm.gray_r(bg_faces * bg_on_data)[kept_indices]

        if colorbar:
            our_cmap = get_cmap(cmap)
            norm = Normalize(vmin=vmin, vmax=vmax)

            # Default number of ticks is 5...
            nb_ticks = 5
            # ...unless we are dealing with integers with a small range
            # in this case, we reduce the number of ticks
            if cbar_tick_format == "%i" and vmax - vmin < nb_ticks:
                ticks = np.arange(vmin, vmax + 1)
            else:
                ticks = np.linspace(vmin, vmax, nb_ticks)

            bounds = np.linspace(vmin, vmax, our_cmap.N)

            if threshold is not None:
                cmaplist = [our_cmap(i) for i in range(our_cmap.N)]
                # set colors to grey for absolute values < threshold
                istart = int(norm(-threshold, clip=True) *
                             (our_cmap.N - 1))
                istop = int(norm(threshold, clip=True) *
                            (our_cmap.N - 1))
                for i in range(istart, istop):
                    cmaplist[i] = (0.5, 0.5, 0.5, 1.)
                our_cmap = LinearSegmentedColormap.from_list(
                    'Custom cmap', cmaplist, our_cmap.N)

            # we need to create a proxy mappable
            proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
            proxy_mappable.set_array(surf_map_faces)
            cax, kw = make_axes(axes, location='right', fraction=.15,
                                shrink=.5, pad=.0,  aspect=10.)
            cbar = figure.colorbar(
                proxy_mappable, cax=cax, ticks=ticks,
                boundaries=bounds, spacing='proportional',
                format=cbar_tick_format, orientation='vertical')
            _crop_colorbar(cbar, cbar_vmin, cbar_vmax)

        p3dcollec.set_facecolors(face_colors)
    
    # Try set background 0
    axes.patch.set_alpha(0)

    if title is not None:
        axes.set_title(title, position=(.5, .95))

    # save figure if output file is given
    if output_file is not None:
        figure.savefig(output_file)
        plt.close(figure)
    else:
        return figure, axes, surf_map_faces


def add_collage_colorbar(figure, ax, smfs, vmin, vmax, 
                         cbar_vmin=None, cbar_vmax=None,
                         multicollage=False,
                         colorbar_params={},
                         **kwargs):

    if 'fraction' in colorbar_params:
        fraction=colorbar_params['fraction']
    else:
        fraction = .5

    if 'shrink' in colorbar_params:
        shrink=colorbar_params['shrink']
    else:
        shrink = .5

    if 'aspect' in colorbar_params:
        aspect=colorbar_params['aspect']
    else:
        aspect = 20

    if 'pad' in colorbar_params:
        pad=colorbar_params['pad']
    else:
        pad = .5

    if 'anchor' in colorbar_params:
        anchor = colorbar_params['anchor']
    else:
        anchor = (0.0, 0.5)

    if 'format' in colorbar_params:
        c_format = colorbar_params['format']
    else:
        c_format = '%.2g'

    if 'cmap' not in kwargs:
        cmap = None
    else:
        cmap = kwargs.pop('cmap')

    if cmap is None:
        cmap = get_cmap(plt.rcParamsDefault['image.cmap'])
    else:
        if isinstance(cmap, str):
            cmap = get_cmap(cmap)

    if 'threshold' in kwargs:
        threshold = kwargs['threshold']
    else:
        threshold = None

    # Color bar
    our_cmap = get_cmap(cmap)

    norm = Normalize(vmin=vmin, vmax=vmax)

    nb_ticks = 5
    ticks = np.linspace(vmin, vmax, nb_ticks)
    bounds = np.linspace(vmin, vmax, our_cmap.N)

    if threshold is not None:
        cmaplist = [our_cmap(i) for i in range(our_cmap.N)]

        # set colors to grey for absolute values < threshold
        istart = int(norm(-threshold, clip=True) *
                     (our_cmap.N - 1))
        istop = int(norm(threshold, clip=True) *
                    (our_cmap.N - 1))
        for i in range(istart, istop):
            cmaplist[i] = (0.5, 0.5, 0.5, 1.)
        our_cmap = LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, our_cmap.N)

    # we need to create a proxy mappable
    proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
    proxy_mappable.set_array(smfs)

    if multicollage:

        cbar = plt.colorbar(
            proxy_mappable, ax=ax, ticks=ticks, spacing='proportional',
            format=c_format, orientation='vertical',
            anchor=anchor, boundaries=bounds,
            fraction=fraction,
            shrink=shrink,
            aspect=aspect,
            pad=pad)

    else:

        left = (ax[0][0].get_position().x0 + ax[0][0].get_position().x1) / 2
        right = (ax[0][1].get_position().x0 + ax[0][1].get_position().x1) / 2
        bot = ax[1][0].get_position().y1
        width = right-left

        # [left, bottom, width, height]
        cbaxes = figure.add_axes([left, bot - (.05 / 3), width, .05])

        cbar = plt.colorbar(
            proxy_mappable, cax=cbaxes, ticks=ticks, spacing='proportional',
            format=c_format, orientation='horizontal', shrink=1, anchor='C')

    _crop_colorbar(cbar, cbar_vmin, cbar_vmax)

    return figure, ax
