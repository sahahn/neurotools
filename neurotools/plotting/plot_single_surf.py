import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colorbar import make_axes
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize, LinearSegmentedColormap

from nilearn.surface import load_surf_data, load_surf_mesh
from nilearn.plotting.img_plotting import (_get_colorbar_and_data_ranges)

import matplotlib.colors as colors
from matplotlib.colors import Normalize, LinearSegmentedColormap


class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...

        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def _crop_colorbar(cbar, cbar_vmin, cbar_vmax):
    """
    crop a colorbar to show from cbar_vmin to cbar_vmax
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
    cbar.ax.set_ylim(cbar.norm(cbar_vmin), cbar.norm(cbar_vmax))
    outline = cbar.outline.get_xy()
    outline[:2, 1] += cbar.norm(cbar_vmin)
    outline[2:6, 1] -= (1. - cbar.norm(cbar_vmax))
    outline[6:, 1] += cbar.norm(cbar_vmin)
    cbar.outline.set_xy(outline)
    cbar.set_ticks(new_tick_locs, update_ticks=True)


def plot_single_surf(surf_mesh, surf_map=None, bg_map=None,
                     hemi='left', view='lateral', cmap=None, colorbar=False,
                     avg_method='mean', threshold=None, alpha='auto',
                     bg_on_data=False, darkness=1, vmin=None, vmax=None,
                     cbar_vmin=None, cbar_vmax=None,
                     title=None, output_file=None, axes=None, figure=None,
                     midpoint=None, dist=6.5, **kwargs):
    """

    Parameters
    ----------
    surf_mesh: str or list of two numpy.ndarray
        Surface mesh geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z coordinates
        of the mesh vertices, the second containing the indices
        (into coords) of the mesh faces.
    surf_map: str or numpy.ndarray, optional.
        Data to be displayed on the surface mesh. Can be a file (valid formats
        are .gii, .mgz, .nii, .nii.gz, or Freesurfer specific files such as
        .thickness, .curv, .sulc, .annot, .label) or
        a Numpy array with a value for each vertex of the surf_mesh.
    bg_map: Surface data object (to be defined), optional,
        Background image to be plotted on the mesh underneath the
        surf_data in greyscale, most likely a sulcal depth map for
        realistic shading.
    hemi : {'left', 'right'}, default is 'left'
        Hemisphere to display.
    view: {'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'},
        default is 'lateral'
        View of the surface that is rendered.
    cmap: matplotlib colormap, str or colormap object, default is None
        To use for plotting of the stat_map. Either a string
        which is a name of a matplotlib colormap, or a matplotlib
        colormap object. If None, matplotlib default will be chosen
    colorbar : bool, optional, default is False
        If True, a colorbar of surf_map is displayed.
    avg_method: {'mean', 'median'}, default is 'mean'
        How to average vertex values to derive the face value, mean results
        in smooth, median in sharp boundaries.
    threshold : a number or None, default is None.
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image, values
        below the threshold (in absolute value) are plotted as transparent.
    alpha: float, alpha level of the mesh (not surf_data), default 'auto'
        If 'auto' is chosen, alpha will default to .5 when no bg_map
        is passed and to 1 if a bg_map is passed.
    bg_on_data: bool, default is False
        If True, and a bg_map is specified, the surf_data data is multiplied
        by the background image, so that e.g. sulcal depth is visible beneath
        the surf_data.
        NOTE: that this non-uniformly changes the surf_data values according
        to e.g the sulcal depth.
    darkness: float, between 0 and 1, default is 1
        Specifying the darkness of the background image.
        1 indicates that the original values of the background are used.
        .5 indicates the background values are reduced by half before being
        applied.
    vmin, vmax: lower / upper bound to plot surf_data values
        If None , the values will be set to min/max of the data
    title : str, optional
        Figure title.
    output_file: str, or None, optional
        The name of an image file to export plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.
    axes: instance of matplotlib axes, None, optional
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `figure, axes = plt.subplots(subplot_kw={'projection': '3d'})`,
        where axes should be passed.).
        If None, a new axes is created.
    figure: instance of matplotlib figure, None, optional
        The figure instance to plot to. If None, a new figure is created.
    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as background map for this plotting function.
    nilearn.plotting.plot_surf_roi : For plotting statistical maps on brain
        surfaces.
    nilearn.plotting.plot_surf_stat_map : for plotting statistical maps on
        brain surfaces.
    """

    # load mesh and derive axes limits
    mesh = load_surf_mesh(surf_mesh)
    coords, faces = mesh[0], mesh[1]
    limits = [coords.min(), coords.max()]

    # set view
    if hemi == 'right':
        if view == 'lateral':
            elev, azim = 0, 0
        elif view == 'medial':
            elev, azim = 0, 180
        elif view == 'dorsal':
            elev, azim = 90, 0
        elif view == 'ventral':
            elev, azim = 270, 0
        elif view == 'anterior':
            elev, azim = 0, 90
        elif view == 'posterior':
            elev, azim = 0, 270
        elif len(view) == 2:
            elev, azim = view
        else:
            raise ValueError('view must be one of lateral, medial, '
                             'dorsal, ventral, anterior, or posterior')
    elif hemi == 'left':
        if view == 'medial':
            elev, azim = 0, 0
        elif view == 'lateral':
            elev, azim = 0, 180
        elif view == 'dorsal':
            elev, azim = 90, 0
        elif view == 'ventral':
            elev, azim = 270, 0
        elif view == 'anterior':
            elev, azim = 0, 90
        elif view == 'posterior':
            elev, azim = 0, 270
        elif len(view) == 2:
            elev, azim = view
        else:
            raise ValueError('view must be one of lateral, medial, '
                             'dorsal, ventral, anterior, or posterior')
    else:
        raise ValueError('hemi must be one of right or left')

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

    # initiate figure and 3d axes
    if axes is None:
        if figure is None:
            figure = plt.figure()
        axes = Axes3D(figure, rect=[0, 0, 1, 1],
                      xlim=limits, ylim=limits)
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
    # control background darkness
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

        # if no vmin/vmax are passed figure them out from data
        if vmin is None:
            vmin = np.nanmin(surf_map_faces)
        if vmax is None:
            vmax = np.nanmax(surf_map_faces)

        # treshold if indicated
        if threshold is None:
            kept_indices = np.arange(surf_map_faces.shape[0])
        else:
            kept_indices = np.where(np.abs(surf_map_faces) >= threshold)[0]

        if midpoint is None:
            surf_map_faces = surf_map_faces - vmin
            surf_map_faces = surf_map_faces / (vmax - vmin)
            norm_object = None

        else:
            norm_object = MidpointNormalize(midpoint=midpoint, vmin=vmin,
                                            vmax=vmax)
            surf_map_faces = norm_object.__call__(surf_map_faces).data

        # multiply data with background if indicated
        if bg_on_data:
            face_colors[kept_indices] = cmap(surf_map_faces[kept_indices])\
                * face_colors[kept_indices]
        else:
            face_colors[kept_indices] = cmap(surf_map_faces[kept_indices])

        if colorbar:
            our_cmap = get_cmap(cmap)

            if midpoint is None:
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = norm_object

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
            proxy_mappable.set_array(surf_map_faces)
            cax, kw = make_axes(axes, location='right', fraction=.1,
                                shrink=.6, pad=.0)
            cbar = figure.colorbar(
                proxy_mappable, cax=cax, ticks=ticks,
                boundaries=bounds, spacing='proportional',
                format='%.2g', orientation='vertical')
            _crop_colorbar(cbar, cbar_vmin, cbar_vmax)

        p3dcollec.set_facecolors(face_colors)

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
                         midpoint=None,
                         multicollage=False,
                         colorbar_params={},
                         **kwargs):

    if 'fraction' in colorbar_params:
        fraction=colorbar_params['fraction']
    else:
        fraction=.5

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

    if midpoint is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = MidpointNormalize(midpoint=midpoint, vmin=vmin,
                                 vmax=vmax)

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
    proxy_mappable.set_array(np.concatenate(smfs))

    if multicollage:

        cbar = plt.colorbar(
            proxy_mappable, ax=ax, ticks=ticks, spacing='proportional',
            format=c_format, orientation='vertical', anchor=anchor,
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
