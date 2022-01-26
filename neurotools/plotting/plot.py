from typing import Type
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .plot_single_surf import plot_single_surf, add_collage_colorbar
from nilearn.plotting import plot_glass_brain, plot_stat_map, plot_roi
from .ref import SurfRef
from ..transform.space import process_space
from scipy.stats import scoreatpercentile
import nibabel as nib
from ..misc.print import _get_print

def _proc_vs(data, vmin, vmax, symmetric_cbar):
    
    # If already set, skip
    if vmin is not None and vmax is not None:
        return vmin, vmax
    
    # Get data as flat array
    flat_data = _collapse_data(data)

    # Small percent of min / max to add to either end
    s = np.nanmax(np.abs(flat_data)) / 25
    
    # Both not set case
    if vmin is None and vmax is None:

        # If not symmetric_cbar, then these value
        # stay fixed as this
        vmin, vmax = np.nanmin(flat_data) - s, np.nanmax(flat_data) + s
        
        # If symmetric need to override either
        # vmin or vmax to be the opposite of the larger
        if symmetric_cbar:
            
            # vmin is larger, override vmax
            if np.abs(vmin) > vmax:
                vmax = -vmin
            
            # vmax is larger, override vmin
            else:
                vmin = -vmax
    
    # If just vmin not set
    if vmin is None:
        
        # If vmax set, vmin not, and symmetric cbar
        # then vmin is -vmax
        if symmetric_cbar:
            vmin = -vmax
        
        # Otherwise, vmin is just the min value in the data
        else:
            np.nanmin(flat_data) - s

    # If vmax not set, same cases as above but flipped
    if vmax is None:

        if symmetric_cbar:
            vmax = -vmin
        else:
            vmax = np.nanmax(flat_data) + s

    return vmin, vmax


def _proc_threshold(data, threshold, percentile=75, rois=False):
    
    # Only proc if left at auto
    if threshold != 'auto':
        return threshold

    # If ROIs then default is .5
    # to not plot rois marked 0
    if rois:
        return .5
    
    # Get flat data
    flat_data = _collapse_data(data)
    
    # Get percentile - not counting any zeros!
    threshold = scoreatpercentile(flat_data[flat_data != 0], per=percentile) - 1e-5
    return threshold


def _collapse_data(data):
    '''Assumes data is in standard {} form.'''
    
    # If directly nifti image
    if isinstance(data, nib.Nifti1Image):
        return np.array(data.get_fdata()).flatten()
    
    # Directly ndarray case
    elif isinstance(data, np.ndarray):
        return np.array(data).flatten()
    
    # Init empty list
    collapsed = []
    
    # Handle passed as dict or list
    # of arrays
    for key in data:

        # Get as item / array
        if isinstance(data, dict):
            item = data[key]
        else:
            item = key

        # Recursive case add
        collapsed.append(_collapse_data(item))

    # Return as concat version
    return np.concatenate(collapsed)


def _get_if_sym_cbar(data, symmetric_cbar, rois=False):
    '''Assumes data is in standard {} form.'''
     
    # If user passed, keep that value
    if symmetric_cbar != 'auto':
        return symmetric_cbar
    
    # If rois, default = False, so return
    if rois:
        return False
    
    # Get data as 1D array
    flat_data = _collapse_data(data)

    # If all positive or negative, assume false
    if np.all(flat_data >= 0) or np.all(flat_data <= 0):
        return False

    # Otherwise, assume is symmetric
    return True

def _add_plots_to_axes(figure, grid, proj_3d, n_cols, n_rows, widths):

    # If passed as single str,  or None, set for all
    if isinstance(proj_3d, str) or proj_3d is None:
        proj_3d = [[proj_3d for _ in range(n_cols)] for _ in range(n_rows)]

    # Otherwise, we use the grid to add_subplots generating axes
    # with each subplot according to the passed proj_3d if it is
    # 3D or not. Use the original number of passed n_rows and n_cols
    axes = []
    for i in range(n_rows):
        for j in range(n_cols):

            if n_rows == 1:
                axes.append(figure.add_subplot(grid[j], projection=proj_3d[i][j]))
            elif len(widths) == 1:
                axes.append(figure.add_subplot(grid[i], projection=proj_3d[i][j]))
            else:
                axes.append(figure.add_subplot(grid[i, j], projection=proj_3d[i][j]))

    return figure, axes


def _setup_fig_axes(widths, heights, 
                    figure=None, subplot_spec=None,
                    get_grid=False, figsize=(10, 10),
                    proj_3d=None, title=None, title_sz=12,
                    title_y=None, colorbar=False, colorbar_params=None,
                    wspace=1, hspace=1):

    # Keep track of the number of original
    # passed cols and rows.
    if widths is None:
        n_cols = 1
    else:
        n_cols = len(widths)
        widths = widths.copy()

    if heights is None:
        n_rows = 1
    else:
        n_rows = len(heights)

    # Proc arg
    if colorbar_params is None:
        colorbar_params = {}

    # Init colorbar ax as None
    colorbar_ax = None

    # If no figure, init
    if figure is None:
        figure = plt.figure(figsize=figsize)

    # If colorbar, add extra width
    if colorbar is True:
        
        if 'cbar_fig_ratio' in colorbar_params:
            widths += [colorbar_params['cbar_fig_ratio']]
        else:
            widths += [.5]

    # Two cases for init'ing grid, either init as nested
    # if subplot_spec, or as base / new
    if subplot_spec is None:
        grid = gridspec.GridSpec(n_rows, len(widths),
                                 wspace=wspace,
                                 hspace=hspace,
                                 width_ratios=widths,
                                 height_ratios=heights)
    else:
        grid =\
            gridspec.GridSpecFromSubplotSpec(n_rows, len(widths),
                                             wspace=wspace,
                                             hspace=hspace,
                                             width_ratios=widths,
                                             height_ratios=heights,
                                             subplot_spec=subplot_spec)
    
    # Next, add the colorbar axis if colorbar, adds by default
    # slash as only option now to the far right
    if colorbar:

        if n_rows == 1:
            colorbar_ax = figure.add_subplot(grid[-1])
        else:
            colorbar_ax = figure.add_subplot(grid[:, -1])
        
        # Make sure to hide axis here
        colorbar_ax.set_axis_off()

    # If a title is passed, also add a special title axis
    if title is not None:
        
        # Colorbar and title case
        if colorbar:
            if n_rows == 1:
                title_ax = figure.add_subplot(grid[:-1])
            else:
                title_ax = figure.add_subplot(grid[:,:-1])

        # Just title
        else:
            title_ax = figure.add_subplot(grid[:])
            
        # Set title and axis off - (hiding)
        title_ax.set_title(title, fontsize=title_sz, y=title_y)
        title_ax.set_axis_off()

    # Optionally, return here, the grid object and figure directly
    if get_grid:
        return figure, grid, colorbar_ax

    # Otherwise add plots to axes
    figure, axes = _add_plots_to_axes(figure, grid, proj_3d,
                                      n_cols, n_rows, widths)

    # Return each, axes instead of grid
    return figure, axes, colorbar_ax


def plot_surf_hemi(data, ref, hemi,
                   surf_mesh='inflated',
                   bg_map='sulc',
                   symmetric_cbar='auto',
                   vmin=None, vmax=None,
                    **kwargs):

    # Proc if sym cbar auto - assume rois False here
    symmetric_cbar = _get_if_sym_cbar(data, symmetric_cbar, rois=False)
    
    # Proc if sym colorbar
    vmin, vmax = _proc_vs(data, vmin, vmax, symmetric_cbar)

    if hemi == 'lh':
        hemi = 'left'
    if hemi == 'rh':
        hemi = 'right'

    # These get surf functions are pretty smart
    # and tolerant to different inputs
    surf_mesh = ref.get_surf(surf_mesh, hemi)
    bg_map = ref.get_surf(bg_map, hemi)

    return plot_single_surf(surf_mesh=surf_mesh,
                            surf_map=data,
                            bg_map=bg_map,
                            hemi=hemi,
                            vmin=vmin,
                            vmax=vmax,
                            **kwargs)


def plot_surf_collage(data, ref=None, surf_mesh='inflated',
                      bg_map='sulc', view='standard',
                      title=None, title_sz=18, title_y=None,
                      vmin=None, vmax=None,
                      cbar_vmin=None, cbar_vmax=None,
                      figsize=(15, 10), symmetric_cbar='auto',
                      figure=None, subplot_spec=None,
                      wspace=-.35, hspace=-.1,
                      colorbar=False, dist=6.5,
                      colorbar_params=None, **kwargs):
    '''
    data should be list with two elements [lh, rh] data.

    view: 'standard' means plot left and right hemisphere lateral
    and medial views in a grid. 
    'fb' / front back, means plot anterior posterior views in a row.
    '''

    # Proc default
    if colorbar_params is None:
        colorbar_params = {}

    # Proc if sym cbar auto - assume rois False here
    symmetric_cbar = _get_if_sym_cbar(data, symmetric_cbar, rois=False)
    
    # Get vmin and vmax based off data + passed args
    vmin, vmax = _proc_vs(data, vmin, vmax, symmetric_cbar)
    smfs = []
    
    # Set params based on requested view
    if view in ['standard', 'default']:
        hemis = ['lh', 'rh', 'lh', 'rh']
        views = ['lateral', 'lateral', 'medial', 'medial']
        widths, heights = [1, 1], [1, 1]

    elif view in ['fb', 'front back', 'ap', 'anterior posterior']:
        hemis = ['lh', 'rh', 'b', 'lh', 'rh']
        views = ['anterior', 'anterior', 'b',
                 'posterior', 'posterior']
        widths, heights = [1, 1, .5, 1, 1], [1]

    elif view in ['f', 'front', 'a', 'anterior']:
        hemis = ['lh', 'rh']
        views = ['anterior', 'anterior']
        widths, heights = [1, 1], [1]

    elif view in ['b', 'back', 'p', 'posterior']:
        hemis = ['lh', 'rh']
        views = ['posterior', 'posterior']
        widths, heights = [1, 1], [1]

    # Setup figres and axes
    figure, axes, colorbar_ax =\
        _setup_fig_axes(widths=widths, heights=heights,
                        figure=figure, subplot_spec=subplot_spec,
                        get_grid=False, figsize=figsize,
                        proj_3d='3d', title=title, title_sz=title_sz, title_y=title_y,
                        colorbar=colorbar, colorbar_params=colorbar_params,
                        wspace=wspace, hspace=hspace)

    # Fill the axes with appropriate values
    for i in range(len(hemis)):

        if hemis[i] == 'lh':
            d = 0
        elif hemis[i] == 'rh':
            d = 1
        else:
            axes[i].patch.set_alpha(0)
            axes[i].set_axis_off()
            continue

        figure, axes[i], smf =\
            plot_surf_hemi(data=data[d], ref=ref, hemi=hemis[i],
                           surf_mesh=surf_mesh, bg_map=bg_map,
                           vmin=vmin, vmax=vmax, view=views[i],
                           colorbar=False, dist=dist,
                           figure=figure, axes=axes[i], **kwargs)
        smfs.append(smf)

    # Set to concat
    smfs = np.concatenate(smfs)
    
    # Add color bar
    if colorbar is True:
        figure, colorbar_ax =\
            add_collage_colorbar(
             figure=figure, ax=colorbar_ax, smfs=smfs,
             vmin=vmin, vmax=vmax,
             cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax,
             multicollage=True,
             colorbar_params=colorbar_params, **kwargs)

    return figure, axes, smfs


def plot_surf_vol_collage(surf, vol,
                          vol_plot_type='glass',
                          cmap='cold_hot',
                          threshold='auto',
                          surf_to_vol_ratio=1,
                          vmin=None, vmax=None,
                          cbar_vmin=None, cbar_vmax=None,
                          figure=None,
                          subplot_spec=None,
                          figsize=(20, 20),
                          title=None, title_sz=18,
                          title_y=None,
                          colorbar=False,
                          symmetric_cbar=False,
                          hspace=0, wspace=0,
                          colorbar_params=None,
                          surf_params=None,
                          vol_params=None,
                          verbose=0, _print=None):

    # Proc defaults
    if colorbar_params is None:
        colorbar_params = {}
    if surf_params is None:
        surf_params = {}
    if vol_params is None:
        vol_params = {}

    # Get verbose print object
    _get_print(verbose, _print=_print)

    # Grab data as list
    data_as_list = [surf[0], surf[1], vol.get_fdata()]

    # Process vmin / vmax from data and passed args
    vmin, vmax = _proc_vs(data_as_list, vmin=vmin, vmax=vmax,
                          symmetric_cbar=symmetric_cbar)

    # Proc threshold if auto
    rois = vol_plot_type == 'roi'
    threshold = _proc_threshold(data_as_list, threshold, rois=rois)

    # Init smfs with min and max of vol data, as
    # these values won't change in
    # in the same way plotting surf vals can.
    smfs = [np.array([np.nanmin(vol.get_fdata()), np.nanmax(vol.get_fdata())])]
    
    # Init settings
    widths = [1]
    heights = [surf_to_vol_ratio] + [1]

    # Setup figures get returned as grid instead of axes
    figure, grid, colorbar_ax =\
        _setup_fig_axes(widths=widths,  heights=heights,
                        figure=figure, subplot_spec=subplot_spec,
                        get_grid=True, figsize=figsize, title=title,
                        title_sz=title_sz, title_y=title_y, colorbar=colorbar,
                        colorbar_params=colorbar_params,
                        wspace=wspace, hspace=hspace)

    # Either multi-index w/ colorbar or just two rows w/o
    if colorbar:
        surf_grid, vol_grid = grid[0, 0], grid[1, 0]
    else:
        surf_grid, vol_grid = grid[0], grid[1]

    # Plot surf collage in the top grid spot
    # The already concatenated smf is returned
    figure, _, smf = plot_surf_collage(surf, 
                                       figure=figure,
                                       subplot_spec=surf_grid,
                                       cmap=cmap,
                                       threshold=threshold,
                                       vmin=vmin,
                                       vmax=vmax,
                                       colorbar=False, # Fixed False
                                       **surf_params)
    
    # Keep track of vals for plotting colorbar
    smfs.append(smf)

    # Plot volume
    vol_ax = figure.add_subplot(vol_grid)

    # Process vol_plot_type
    if vol_plot_type == 'glass':
        vol_plot_func = plot_glass_brain
    elif vol_plot_type == 'stat':
        vol_plot_func = plot_stat_map
    elif vol_plot_type == 'roi':
        vol_plot_func = plot_roi
    else:
        raise RuntimeError('vol_plot_type must be one of "glass", "stat" or "roi".')
    
    # Handle funny case where plotting glass
    # but the plotted data is symmetric_cbar and plot abs not specified
    if vol_plot_type == 'glass' and 'plot_abs' not in vol_params:

        # If symmetric cbar, then we want
        # glass to not plot abs
        if symmetric_cbar:
            vol_params['plot_abs'] = False
            _print('Plotting glass brain with plot_abs=False.', level=1)
 
    # Call base plot function
    vol_plot_func(vol,
                  figure=figure, axes=vol_ax, 
                  cmap=cmap, threshold=threshold,
                  vmax=vmax, colorbar=False,
                  **vol_params)

    # Set to concat
    smfs = np.concatenate(smfs)

    # Add color bar
    if colorbar is True:
        figure, colorbar_ax =\
            add_collage_colorbar(
             figure=figure, ax=colorbar_ax, smfs=smfs,
             vmin=vmin, vmax=vmax,
             cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax,
             multicollage=True,
             colorbar_params=colorbar_params, cmap=cmap,
             threshold=threshold)

    return figure, grid, smfs

## Smart Plot Functions ##

def _proc_ref_arg_defaults(ref, surf_mesh, bg_map, darkness):

    # Allow any user passed args to override space ref defaults here
    if surf_mesh is None:
        surf_mesh = ref.surf_mesh
    if bg_map is None:
        bg_map = ref.bg_map
    if darkness is None:
        darkness = ref.darkness

    return surf_mesh, bg_map, darkness


def _load_data_and_ref(data, space=None, hemi=None, _print=None):
    
    # Process the data + space - returns data as dict
    data, space = process_space(data, space, hemi, _print=_print)
    
    # If no space, i.e., just sub, return no Ref
    if space is None:
        return data, None

    # Otherwise generate SurfRef with defaults
    ref = SurfRef(space=space)
    
    # If assumed native space, then assume user will pass mesh + bg_map
    if space == 'native':
        ref.surf_mesh = None
        ref.bg_map = None
        ref.darkness = 1
    
    # fs_LR space defaults
    elif 'fs_LR' in space:
        ref.surf_mesh = 'very_inflated'
        ref.bg_map = 'sulc'
        ref.darkness = .5
    
    # Freesurfer spaces defaults
    else:
        ref.surf_mesh = 'inflated'
        ref.bg_map = 'sulc'
        ref.darkness = 1

    return data, ref


def _proc_avg_method(data, rois, avg_method):

    # TODO need volumetric case here?

    # If rois force avg method to median
    if rois:
        return 'median'
    
    # Otherwise set based on if we think we are
    # plotting roi values on a surface
    if avg_method == 'default':
        
        flat_data = _collapse_data(data)

        # Get flat non-zero
        flat_non_zero = flat_data[flat_data != 0]
        n_unique = len(np.unique(flat_non_zero))

        # If more than 5% data points different
        if n_unique / len(flat_non_zero) > 0.05:
            avg_method = 'mean'
        else:
            avg_method = 'median'

    return avg_method

def _proc_colorbar(colorbar, rois):

    if colorbar == 'default':
        colorbar = True
        
        if rois:
            colorbar = False
    
    return colorbar


def _prep_base_auto_defaults(data_as_list, rois,
                             symmetric_cbar, threshold,
                             cmap, colorbar, avg_method):
            
    # Collapse data
    flat_data = _collapse_data(data_as_list)
    
    # Process automatic symmetric_cbar
    symmetric_cbar = _get_if_sym_cbar(flat_data,
                                      symmetric_cbar,
                                      rois=rois) 

    # Proc threshold if auto
    threshold = _proc_threshold(flat_data, threshold, rois=rois)
    
    # Get cmap based on passed + if rois or not + if sym
    cmap = _proc_cmap(cmap, rois, symmetric_cbar)
    
    # Proc colorbar option - yes if not rois
    colorbar = _proc_colorbar(colorbar, rois)

    # Proc avg method default
    avg_method = _proc_avg_method(flat_data, rois, avg_method)

    # Return
    return symmetric_cbar, threshold, cmap, colorbar, avg_method


def _prep_auto_defaults(data, space, hemi, rois,
                        symmetric_cbar, threshold,
                        cmap, colorbar, avg_method, _print):

    # Fine passing already proc'ed whatever here
    data, ref = _load_data_and_ref(data, space=space, hemi=hemi, _print=_print)

    # Process
    symmetric_cbar, threshold, cmap, colorbar, avg_method =\
        _prep_base_auto_defaults(data, rois,
                                 symmetric_cbar, threshold,
                                 cmap, colorbar, avg_method)

    return data, ref, symmetric_cbar, threshold, cmap, colorbar, avg_method


def _plot_surfs(data, space=None, hemi=None, surf_mesh=None,
                bg_map=None, rois=False, cmap='default',
                bg_on_data=.25, darkness=None,  avg_method='default',
                wspace=-.35, hspace=-.1, alpha=1,
                threshold='auto', symmetric_cbar='auto', 
                colorbar='default', _print=None, **kwargs):

    # Process default surface and plotting values
    data, ref, symmetric_cbar, threshold, cmap, colorbar, avg_method =\
        _prep_auto_defaults(data, space, hemi, rois,
                            symmetric_cbar, threshold,
                            cmap, colorbar, avg_method, _print=_print)
 
    # If user-passed - update params
    surf_mesh, bg_map, darkness =\
        _proc_ref_arg_defaults(ref, surf_mesh, bg_map, darkness)

    # Both hemi's passed case.
    if 'lh' in data and 'rh' in data:
        return plot_surf_collage(data=[data['lh'], data['rh']],
                                 ref=ref,
                                 surf_mesh=surf_mesh,
                                 bg_map=bg_map,
                                 cmap=cmap,
                                 avg_method=avg_method,
                                 threshold=threshold,
                                 symmetric_cbar=symmetric_cbar,
                                 alpha=alpha,
                                 bg_on_data=bg_on_data,
                                 darkness=darkness,
                                 wspace=wspace, hspace=hspace,
                                 colorbar=colorbar, **kwargs)

    # Just lh/ rh cases
    if 'lh' in data:
        hemi_data, hemi = data['lh'], 'left'
    if 'rh' in data:
        hemi_data, hemi = data['rh'], 'right'
    
    # Plot
    return plot_surf_hemi(data=hemi_data,
                          ref=ref,
                          hemi=hemi,
                          surf_mesh=surf_mesh,
                          bg_map=bg_map,
                          cmap=cmap,
                          avg_method=avg_method,
                          threshold=threshold,
                          symmetric_cbar=symmetric_cbar,
                          alpha=alpha,
                          bg_on_data=bg_on_data,
                          darkness=darkness,
                          colorbar=colorbar,
                          **kwargs)

def _sort_kwargs(kwargs):

    # Surface specific
    surf_args = ['alpha', 'bg_map',
                 'surf_mesh', 'darkness',
                 'view', 'dist']
    surf_params = {key: kwargs[key] for key in surf_args if key in kwargs}
    
    # Special cases
    if 'surf_hspace' in kwargs:
        surf_params['hspace'] = kwargs['surf_hspace']

    # If also passed directly
    # Give priority to any if passed in surf_params vs. kwargs
    if 'surf_params' in kwargs:
        surf_params = {**surf_params, **kwargs['surf_params']}
    
    # Volume specific
    vol_args = ['resampling_interpolation', 'plot_abs',
                'black_bg', 'display_mode', 'annotate', 'cut_coords',
                'bg_img', 'draw_cross', 'dim', 'view_type', 'linewidths']
    vol_params = {key: kwargs[key] for key in vol_args if key in kwargs}

    # Special case
    if 'vol_alpha' in kwargs:
        vol_params['alpha'] = kwargs['vol_alpha']

    # If also passed directly
    # Give priority to any in passed vol params
    if 'vol_params' in kwargs:
        vol_params = {**vol_params, **kwargs['vol_params']}
    
    # For color bar params
    colorbar_params = _sort_colorbar_kwargs(colorbar_params=None, **kwargs)

    return surf_params, vol_params, colorbar_params

def _sort_colorbar_kwargs(colorbar_params=None, **kwargs):

    # Handle if not passed
    if colorbar_params is None:
        if 'colorbar_params' in kwargs:
            colorbar_params = kwargs.pop('colorbar_params')
        else:
            colorbar_params = {}

    # Colorbar specific args
    colorbar_args = ['fraction', 'shrink', 'aspect',
                     'pad', 'anchor', 'format', 'cbar_fig_ratio']
  
    # Pop these arguments
    kwargs_colorbar_params = {key: kwargs.pop(key) for key in colorbar_args if key in kwargs}

    # Combine
    colorbar_params = {**colorbar_params, **kwargs_colorbar_params}

    return colorbar_params

def _proc_cmap(cmap, rois, symmetric_cbar):
    
    # Keep user passed if not user passed
    if cmap != 'default':
        return cmap
    
    # If plotting rois
    if rois:
        return 'prism'
    
    # If not symmetric, then just do Reds
    if symmetric_cbar is False:
        return 'Reds'

    # Last case is symmetric cbar
    return 'cold_hot'


def _plot_surfs_vol(data, space=None, hemi=None,
                    rois=False, cmap='default',
                    vol_plot_type='glass',
                    surf_mesh=None, bg_map=None,
                    colorbar='default', symmetric_cbar='auto',
                    darkness=None, vmin=None, vmax=None,
                    cbar_vmin=None, cbar_vmax=None,
                    figure=None, subplot_spec=None,
                    figsize='default', title=None,
                    title_sz=18, title_y=None,
                    hspace='default', wspace=-.2, alpha=1,
                    avg_method='default', threshold='auto',
                    surf_to_vol_ratio='default', surf_wspace='default',
                    bg_on_data=.25, _print=None, **kwargs):

    # Process default surface and plotting values
    data, ref, symmetric_cbar, threshold, cmap, colorbar, avg_method =\
        _prep_auto_defaults(data, space, hemi, rois,
                            symmetric_cbar, threshold,
                            cmap, colorbar, avg_method, _print=_print)
    
    # If user-passed
    surf_mesh, bg_map, darkness = _proc_ref_arg_defaults(ref, surf_mesh, bg_map, darkness)

    # Set default anchor value
    if 'anchor' not in kwargs:
        kwargs['anchor'] = (0, .6)

    # Set specific plot settings if glass or not
    # due to differences in sizes
    if vol_plot_type == 'glass':

        if surf_to_vol_ratio == 'default':
            surf_to_vol_ratio = 1.1
        if hspace == 'default':
            hspace = -.2
        if surf_wspace == 'default':
            surf_wspace = -.125

    # Roi's / plot stat volume case
    else:
        if surf_to_vol_ratio == 'default':
            surf_to_vol_ratio = .9
        if hspace == 'default':
            hspace = -.25
        if surf_wspace == 'default':
            surf_wspace = -.2
    
    # Set default figsize based on if colorbar
    if figsize == 'default':

        if colorbar:
            figsize = (12, 12)
        else:
            figsize = (9, 12)

    # Sort kwargs into categories
    surf_params, vol_params, colorbar_params = _sort_kwargs(kwargs)

    # If plotting rois, force values to plot roi specific
    if rois:
        vol_plot_type = 'roi'

    # Add surface specific to surf_params dict
    surf_params['surf_mesh'], surf_params['bg_map'] = surf_mesh, bg_map
    surf_params['darkness'], surf_params['alpha'] = darkness, alpha
    surf_params['ref'], surf_params['avg_method'] = ref, avg_method
    surf_params['wspace'], surf_params['bg_on_data'] = surf_wspace, bg_on_data
    
    # Pass arguments to plot surf vol, and return
    return plot_surf_vol_collage(surf=[data['lh'], data['rh']],
                                 vol=data['vol'], vol_plot_type=vol_plot_type,
                                 cmap=cmap, threshold=threshold, vmin=vmin, vmax=vmax,
                                 cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax,
                                 figure=figure, subplot_spec=subplot_spec,
                                 figsize=figsize, title=title, title_sz=title_sz, title_y=title_y,
                                 colorbar=colorbar, symmetric_cbar=symmetric_cbar,
                                 hspace=hspace, wspace=wspace, colorbar_params=colorbar_params,
                                 surf_params=surf_params, vol_params=vol_params,
                                 surf_to_vol_ratio=surf_to_vol_ratio, _print=_print)

def plot_volume(vol,
                vol_plot_type='glass',
                cmap=None,
                colorbar='default',
                threshold='auto',
                vmax=None,
                figure=None,
                axes=None,
                **kwargs):

    # Base stat cases
    rois = False
    if vol_plot_type == 'glass':
        vol_plot_func = plot_glass_brain
    elif vol_plot_type == 'stat':
        vol_plot_func = plot_stat_map
    
    # ROI case
    elif vol_plot_type == 'roi':
        vol_plot_func = plot_roi
        rois = True

    # Proc threshold if auto
    threshold = _proc_threshold(vol, threshold, rois=rois)
    
    # Add or don't add colorbar
    colorbar = _proc_colorbar(colorbar, rois=rois)
    
    # Call vol plot function
    vol_plot_func(vol, figure=figure,  axes=axes,
                  cmap=cmap, threshold=threshold,
                  vmax=vmax, colorbar=colorbar, **kwargs)
    
    # Set if used by upper level functions to return
    vol_smfs = [np.array([np.nanmin(vol.get_fdata()), np.nanmax(vol.get_fdata())])]

    return figure, axes, vol_smfs


def _setup_auto_plot(data, space=None, hemi=None, verbose=0, **kwargs):

    # Get verbose object
    _print = _get_print(verbose=verbose)
    
    # Load / perform initial auto-detection of data
    data, _ = _load_data_and_ref(data, space=space, hemi=hemi, _print=_print)

    # If includes surface data -
    # Get the unique values
    if 'lh' in data:
        unique_vals = np.unique(data['lh'])
    elif 'rh' in data:
        unique_vals = np.unique(data['rh'])
    # If no surface data present, subcort case
    else:
        unique_vals = np.unique(data['vol'].get_fdata())

    # If all of the data points are interger-like, assume
    # we are plotting a parcellation and not stat data
    if all([float(u).is_integer() for u in unique_vals]):
        
        # Just surface or surfaces case
        if 'vol' not in data:
            return data, _plot_surfs, {'space': space,
                                       'hemi': hemi,
                                       'rois': True,
                                       '_print': _print,
                                       **kwargs}
        
        # Just volume case
        elif 'lh' not in data and 'rh' not in data:
           
            # Override vol_plot_type - force roi
            return data['vol'], plot_volume, {'vol_plot_type': 'roi', **kwargs}
            
        # Last case is vol / surf collage

        return data, _plot_surfs_vol, {'space': None,
                                       'hemi': None,
                                       'rois': True,
                                       '_print': _print,
                                       **kwargs}
    
    # Otherwise - stat values

    # Just surface or surfaces case
    if 'vol' not in data:
        return data, _plot_surfs, {'space': space,
                                   'hemi': hemi,
                                   'rois': False,
                                   '_print': _print,
                                   **kwargs}

    # Just volume case
    elif 'lh' not in data and 'rh' not in data:
        return data['vol'], plot_volume, kwargs

    # Last case is vol / surf collage
    return data, _plot_surfs_vol, {'space': None,
                                   'hemi': None,
                                   'rois': False,
                                   '_print': _print,
                                   **kwargs}


def plot(data, space=None, hemi=None, verbose=0, returns=False, **kwargs):
    '''The most automated magic plotting function avaliable,
    used to plot a wide range of neuroimaging data (volumes / surfs).

    Parameters
    -----------
    data : str, array, dict, ect...
        The data in which to plot, either statistical map or parcellaction,
        as broadly either a single surface, collage of surfaces, a single
        volume, or collage with surfaces and volume.
    
        Data can be passed in many ways:

        - As array-like input representing a single surface hemisphere,
          concatenated hemispheres, concatenated hemispheres + flattened sub-cortical
          values, or just flattened volumetric values.

        - As a str, representing the file location with saved values
          for any of the above array options. A large number of different
          file formats are accepted.

        - As a dictionary. If passed as a dictionary, this
          allows the user most control over how data is specified.
          Dictionary keys must be one or more of 'lh', 'rh' and 'vol',
          where values can be either array-like, file locations or nibabel
          objects. Note that if you are trying to plot data in native
          subject space, it must be passed in this dictionary style, along
          with, if a surface, a proper value for surf_mesh.

        - As a list or array-like of length 3, indicating internally
          that the passed values represent in the first index the file location
          or array values of a left hemisphere surface, and in the second index,
          the right hemisphere surface.

        - As a list or array-like of length 3, indicating internally
          that the first two passed index are left and right hemisphere
          surface data, and the last is subcortical / volumetric data, all to
          be plotted together in a collage.

        - As a :class:`Nifti1Image<nibabel.nifti1.Nifti1Image>` to be plotted volumetrically.

    space : None or str, optional
        This argument defines the "space" in which
        surface data is plotted. If left as default (None),
        the space in which the data to plot is
        in will be automatically inferred,
        otherwise this parameter can be manually set,
        overriding the automatically detected choice.
        Current supported options are:

        - 'fsaverage' : freesurfer average standard space.

        - 'fsaverage5' : freesurfer5 downsampled average standard space.

        - 'fsaverage4' : freesurfer3 downsampled average standard space.

        - 'fsaverage3' : freesurfer3 downsampled average standard space.

        - '41k_fsaverage' : Matched to CIVET 41k vertex freesurfer space.

        - '4k_fs_LR' :  HCP standard 4k vertex space. 

        - '8k_fs_LR' :  HCP standard 8k vertex space. 

        - '32k_fs_LR' : HCP standard 32k vertex space. 

        - '59k_fs_LR' : HCP standard 59k vertex space. 

        - '164k_fs_LR' : HCP standard 164k vertex space. Note, while this space has the same number of vertex as fsaverage space, it is not the same.

        - 'civet' : Standard space with 41k vertex used by CIVET.

        - 'native' : This is set if the passed number of values don't correspond to any of the saved defaults, and refers to that the space to plot is in some non-standard space.

        ::

            default = None

    hemi : None, 'lh' or 'rh', optional
        This parameter is only used when plotting
        any surface data. Further, this parameter
        is only relevant the surface data being plotted represents
        a single hemispheres data.
        
        If left as default, None,
        then this option will be automatically set to plot
        left hemisphere data.

        ::

            default = None

    verbose : int, optional
        Plotting includes a many, many of parameters,
        which is why this function is helpful as it automates a
        large number of choices. That said, it can helpful
        to be aware of what choices are being made, which is
        what this verbose parameter controls. The following
        levels are made avaliable:

        - <0 : Nothing, not even warnings are shown.
        - 0 : Only warnings are shown.
        - 1 : Information on which steps are automatically decided are shown.
        - >1 : Automatic choices are shown as well as additional helper text.

    returns : bool, optional
        If the figure, axis / grid and smfs should be returned from the base function
        calls or not.

        ::

            default = False

    kwargs : keyword arguments
        There are number of different plotting specific arguments
        that can optionally be modified. The default values for these arguments
        may change also depending on different auto-detected settings, i.e.,
        if plotting a single surface vs. surface collage, or if plotting
        a volumetric image. The remaining parameters below are these kwargs.

        - **surf_mesh** : str or array
        
          A str indicator specifying a valid surface mesh,
          with respect to the current space, or a two dimensional array
          with information about the coordinates and vertex-faces
          in which to plot the data on. The default for
          freesurfer spaces is 'inflated' and for fs_LR spaces
          is 'very_inflated'. If you wish to plot surface
          data in native subject space, then this argument must be
          supplied with a loaded 2D array / valid mesh, and data
          must be passed in dictionary style.

          This parameter is only relevant when plotting surfaces.

        - **bg_map** : str or 2D array
        
          This argument specifies a background map to be used when plotting. This
          should be passed as either a str indicator specifying a valid file
          within the current surface space, or as a valid array of values,
          again, with respect to the current space. This map is plotted in greyscale
          underneath the data points, often used for realistic shading.
          The default for when plotting is 'sulc'.

        - **cmap** : str or :class:`matplotlib.colors.Colormap`
          
          This should be pass as an instance
          of :class:`matplotlib.colors.Colormap` or str
          representing the name of a matplotlib colormap. This will be the color map
          in which the values are plotted according to. When plotting surface
          parcellations, the default cmaps are 'prism' if plotting rois, 'Reds'
          if plotting not symmetric statistical maps, and 'cold_hot' if plotting
          symmetric statistical maps (e.g., values above and below 0).

        - **vol_plot_type** : {'glass', 'stat', 'roi'}
        
          This parameter control the type of volumetric plot in which to generate,
          with valid values as one of 'glass', 'stat' and 'roi'. By default,
          if detected to be plotting a parcellation, the value 'roi' is used.
          Otherwise, the default volumetric plotting type is 'glass'.
          The corresponding back-end nilearn functions used are:

          - 'glass': :func:`nilearn.plotting.plot_glass_brain`
          - 'stat': :func:`nilearn.plotting.plot_stat_map`
          - 'roi': :func:`nilearn.plotting.plot_roi`

          *This parameter is only used when plotting volumetric data.*

        - **bg_on_data** : bool or float
          If True, and a bg_map is specified,
          the data to plot is multiplied by the background image, 
          so that e.g. sulcal depth is visible beneath the surf_data.

          If passed as a float value, then that will trigger multiplying
          the data to plot by the background image times the passed bg_on_data
          value. So for example passing True is equivalent to passing 1 and passing
          False is the same as passing 0.
          
          This allows for fine tuning how much the background image
          is melded with the data shown. The default value is .25.

          *This parameter is only used when plotting surfaces.*

        - **view** : str
        
          If plotting a single surface hemisphere, this
          parameter must be one of:

            {'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'}
          
          Where 'lateral' is the default view if none is set.

          If instead plotting a collage of surface views, then
          valid view parameters are one of:

            {'standard', 'front back', 'front', 'back'}
          
          which correspond to different preset collections of surface views.
          The default if not specified in this case is 'standard'.
        
          Note: If plotting a collage of surface and volumetric views,
          it is reccomended to keep the default collage 'standard' view,
          as the other views have not yet been properly formatted yet.

        - **colorbar** : bool

          This option dictates if a colorbar should be added
          along with the requested plot, regardless of type.
          By default a colorbar will be added (True) unless
          a parcellation is being plotted, in which the default is
          to not add a colorbar (False).

          There are a number of extra more detailed colorbar style
          kwargs, used in certain cases, with certain plots to
          customize the look and size of the colorbar. See below.

        - **avg_method** : 'mean', 'median', 'min', 'max' or  custom function

          This option defines how to average vertex values to
          derive face values, and is only used when plotting surfaces.

          - 'mean': results in smooth boundaries

          - 'median': results in sharp boundaries

          - 'min' or 'max': for sparse matrices

          - custom function: You can also pass a custom function

          If plotting roi's the default will be 'median'. Otherwise, the default
          will be set based on the ratio between the number of unique values
          to data points. So if more than 5% of data points are unique, 'mean'
          will be used, otherwise 'median' will be used. This is to try and accurately detect
          when parcellations are being plotted.

        - **alpha** : float or 'auto'

          This parameter is only used when plotting surface data. It
          refers to the alpha level of the mesh. By default, this
          is left as 'auto', which will default to .5 if no bg_map
          and 1 if a bg_map is passed. This parameter is just multiplied by
          the face colors before plotting.

          Note: this keyword is used for plotting surfaces, though alpha is
          also used as a parameter for plotting the volumetric glass brain.
          To access this parameter for the volumetric function, see the
          `vol_alpha` parameter.

        - **vol_alpha** : float between 0 and 1
        
          This controls the alpha transparency only when plotting
          volumetric data according to the vol_plot_type='glass' option.
          The default value is 0.7.
        
        - **darkness** : float between 0 and 1

          This parameter specified the darkness of the background image,
          when plotting a single or multiple surfaces. Where a value of 1
          indicates that the original values are used, and .5 would mean the
          values are halved. If plotting in an fsaverage space the default value is 1,
          if plotting in an fs_LR space, the default value is .5. 

        - **symmetric_cbar** : bool or 'aut'
        
            Specifies whether the colorbar should range from -vmax to vmax or from vmin to vmax.
            Setting to 'auto' will select the latter if the range of
            the whole image is either positive or negative. There are some
            other automatic cases, but in general this parameter
            can just be used to override any automatic choices.

        - **threshold** : float, None or 'auto'

          If None, the image is not thresholded. If a float,
          or other numberic value, that value will be used to threshold
          the image. This threshold is treated as an absolute value in
          the case of a symmetric_cbar. If 'auto' is passed,
          it will attempt to automatically set the threshold to
          a reasonable value. 

          The default is typically 'auto', but in some cases changes,
          for example if plotting a parcellation the default value will be
          None.s

        - **figsize** : 'default' or (int, int)

          This parameter is used when axes are not passed, and
          a new figure is being generated. It represents
          the size of the underlying matplotlib figure to plot on.
          The default value for this function varies based on what type of
          plot is being made (e.g., not the same for surface collage vs. single surface).

        - **wspace** : float

          This parameter refers to the width spacing between items at the top
          level of whatever collage is being plotted. So, if plotting only a single
          surface or volume, this parameter will be ignored. Otherwise,
          for example if plotting a collage of 4 surface views, then this
          parameter will control the amount of horizontal space between each
          brain view. Instead, if plotting a combined collage of surface views
          and volumetric views, then the top level of the collage that this parameter
          controls the spacing of is the set of all surface views, all volumetric views
          and colorbar. In this case, to override values between for example just the
          surface views, you would have to use special extra keyword 'surf_wspace',
          which let's you set both parameters if desired.

          The default values hover around 0, and vary a great deal based on the type
          of plot and requested view.

        - **hspace** : float

          This parameter refers to the height spacing between items at the top
          level of whatever collage is being plotted. So, if plotting only a single
          surface or volume, this parameter will be ignored. Otherwise,
          for example if plotting a collage of 4 surface views, then this
          parameter will control the amount of vertical space between each
          brain view. Instead, if plotting a combined collage of surface views
          and volumetric views, then the top level of the collage that this parameter
          controls the spacing of is the set of all surface views, all volumetric views
          and colorbar. In this case, to override values between for example just the
          surface views, you would have to use special extra keyword 'surf_hspace',
          which let's you set both parameters if desired.

          The default values hover around 0, and vary a great deal based on the type
          of plot and requested view.

        - **surf_wspace** : float
          
          See 'wspace'. This parameter is only used when there
          are multiple levels of collage and further control is still
          required for specifically the nested collage of surface plots.

        - **surf_hspace** : float
          
          See 'hspace'. This parameter is only used when there
          are multiple levels of collage and further control is still
          required for specifically the nested collage of surface plots.

        - **surf_to_vol_ratio** : float or 'default'

          In the case of plotting a collage of both surfaces and
          volumetric data, this parameter can optionally be set.
          It defines the size ratio between the grouping of surface collages
          and the volumetric plots. If left as 'default', will
          be set to either .9 in base plotting stat volume case
          and 1.1 when plotting glass brain.

    Notes
    -----------
    The creation of carefully constructed multi-figures can be a little tricky,
    which is why it is helpful that so many of the default values have been set. That
    said, it is likely the interested user could do better than the defaults with
    say for example plotting a multi-figure surface + volume collage. In this instance,
    if they wanted to tweak the positioning of the different sub figures relative to
    each other, they could make use of the following parameters:

    - hspace
    - wspace
    - surf_hspace
    - surf_wspace
    - surf_to_vol_ratio
    - cbar_fig_ratio
    - figsize

    '''

    # Setup
    data, func, args = _setup_auto_plot(data, space=space, hemi=hemi,
                                        verbose=verbose, **kwargs)

    # Call func
    r = func(data, **args)

    # Optionally return Figure + whatever returned by sub calls
    if returns:
        return r

    return None


def _get_is_roi(args):
    
    # Vol case
    if 'vol_plot_type' in args:
        if args['vol_plot_type'] == 'roi':
            return True
    
    # Surf case
    elif 'rois' in args:
        if args['rois']:
            return True
        
    return False

def _get_data_func_types(data):
    
    data_func_types = {}
    data_as_list = []
    is_rois = []
    
    for key in list(data):
        loaded, func, args = _setup_auto_plot(data[key], space=None,
                                              hemi=None, verbose=0)
        data_func_types[key] = func.__name__
        data_as_list.append(loaded)
        is_rois.append(_get_is_roi(args))

    return data_func_types, data_as_list, is_rois


def _add_grid_wh(sz, layout_params, max_cols=2):
    
    # Return single row, w/ length sz
    if sz < max_cols:
        ws = [1 for _ in range(sz)]
        hs = [1]
    
    # Otherwise calc as grid, w/ leave empty any missing
    else:
        n_rows = (sz+1) // max_cols
        ws = [1 for _ in range(max_cols)]
        hs = [1 for _ in range(n_rows)]
        
    # Add to layout params
    layout_params['widths'] = ws
    layout_params['heights'] = hs
    
    # Return n_rows, n_cols for conv.
    return len(ws), len(hs)

def _get_def_layout_params(unique_func_types, data_keys, colorbar, colorbar_params):

    # Init dict of layout params
    layout_params = {}

    # All same cases
    if len(unique_func_types) == 1:
        
        # If all same, then get single type
        single_type = unique_func_types[0]

        # TODO change max cols maybe based on single_type
        
        # Gen base grid based on number of things to plot
        n_cols, n_rows = _add_grid_wh(len(data_keys), layout_params, max_cols=2)

        # Collage of just surfaces
        if single_type == '_plot_surfs':
        
            # Base colorbar vs. no cbar settings, first default None case
            layout_params['title_y'] = None
            layout_params['sub_title_y'] = .9 - ((n_rows-1)/100)
            layout_params['title_sz'] = 27
            
            # Colorbar case
            if colorbar:
                layout_params['title_y'] = .95
                layout_params['sub_title_y'] = .83
                layout_params['title_sz'] = 24
            
            # Set fig size
            layout_params['figsize'] = (10*n_cols, 10*n_rows)
            
            # Adjust title sz
            layout_params['title_sz'] += (n_rows-1) * 3
            
            # Other layout settings
            layout_params['wspace'] = .15
            layout_params['sub_wspace'] = .1
            
            if colorbar:
                if n_rows == 1:
                    layout_params['hspace'] = 0
                    layout_params['sub_hspace'] = -.5
                elif n_rows == 2:
                    layout_params['hspace'] = -.35
                    layout_params['sub_hspace'] = -.58
                else:
                    raise RuntimeError()
           
            else:
                if n_rows == 1:
                    layout_params['hspace'] = 0
                    layout_params['sub_hspace'] = -.31
                elif n_rows == 2:
                    layout_params['hspace'] = -.185
                    layout_params['sub_hspace'] = -.390
                elif n_rows > 2:
                    layout_params['hspace'] = -.185 - ((n_rows - 2)  * .025)
                    layout_params['sub_hspace'] = -.390 - ((n_rows - 2) * .02)

            
            # Only set fraction if one row
            if n_rows == 1:
                colorbar_params['fraction'] = 1

        # Surf and vol case
        elif single_type == '_plot_surfs_vol':
            
            layout_params['title_y'] = .95
            layout_params['sub_title_y'] = .83
            layout_params['title_sz'] = 24

    return layout_params

def meta_collage(data, title=None, sub_titles=True,
                 colorbar='default', verbose=0, vmin=None,
                 vmax=None, colorbar_params=None, threshold='auto',
                 symmetric_cbar='auto', cmap='default', 
                 avg_method='default', sub_kwargs=None, **kwargs):
    
    # Check explicit sub kwargs
    if sub_kwargs is None:
        sub_kwargs = {}
    
    # Check initial data
    data_func_types, data_as_list, is_rois = _get_data_func_types(data)
    unique_func_types = list(set(data_func_types.values()))
    data_keys = list(data)
    
    # Catch error
    if len(set(is_rois)) > 1 and colorbar:
        raise RuntimeError('Cant plot mix of rois and stat maps with colorbar.')
    
    # Passes True if all True
    rois = all(is_rois)
    
    # Collapse data
    flat_data = _collapse_data(data_as_list)
    
    # Process some base auto args
    symmetric_cbar, threshold, cmap, colorbar, avg_method =\
        _prep_base_auto_defaults(flat_data, rois,
                                 symmetric_cbar, threshold,
                                 cmap, colorbar, avg_method)

    
    # Process vmin / vmax from data and passed args
    vmin, vmax = _proc_vs(data_as_list, vmin=vmin, vmax=vmax,
                          symmetric_cbar=symmetric_cbar)
    
    # Proc default / add params
    colorbar_params = _sort_colorbar_kwargs(colorbar_params, **kwargs)

    # Get default layout params
    layout_params = _get_def_layout_params(unique_func_types, data_keys, colorbar, colorbar_params)
    
    # Need to check if any user over rides in layout params
    for key in layout_params:
        
        # If user passed, favor that
        if key in kwargs:
            layout_params[key] = kwargs.pop(key)
            
    # Fill in sub kwargs
    def_sub_kwargs = {}
    def_sub_kwargs['wspace'] = layout_params.pop('sub_wspace')
    def_sub_kwargs['hspace'] = layout_params.pop('sub_hspace')
    def_sub_kwargs['title_y'] = layout_params.pop('sub_title_y')
    
    if 'sub_title_sz' in kwargs:
        def_sub_kwargs['title_sz'] = kwargs.pop('sub_title_sz')
        
    # Idea here is to give priority to anything passed
    # explicitly with sub_kwargs, then kwargs, then defaults
    sub_kwargs = {**kwargs, **def_sub_kwargs, **sub_kwargs}
                
    # Setup base collage settings / grid spec
    figure, grid, colorbar_ax = _setup_fig_axes(get_grid=True,
                                                title=title,
                                                colorbar=colorbar,
                                                colorbar_params=colorbar_params,
                                                **layout_params)
    
 
    # Add sub plots
    smfs = []
    for i in range(len(layout_params['heights'])):
        for j in range(len(layout_params['widths'])):
            key = data_keys[i+j]
            
            # Optional sub title
            if isinstance(sub_titles, bool):
                if sub_titles:
                    sub_title = str(key)
                else:
                    sub_title = None

            # Otherwise if list
            elif isinstance(sub_titles, list):
                sub_title = sub_titles[i+j]
            
            # Or dict
            elif isinstance(sub_titles, dict):
                sub_title = sub_titles[key]
            
            # Last case is just None
            else:
                sub_title = None

            # Call plot
            _, _, smfz = plot(data[key],
                              figure=figure,
                              subplot_spec=grid[i, j],
                              colorbar=False,
                              title=sub_title,
                              symmetric_cbar=symmetric_cbar,
                              threshold=threshold,
                              cmap=cmap, vmin=vmin, vmax=vmax,
                              avg_method=avg_method, verbose=verbose,
                              returns=True,
                              **sub_kwargs)
            
            
            smfs.append(smfz)
            
    # Set to concat
    smfs = np.concatenate(smfs)
    
    # Add collage color bar
    if colorbar:
        add_collage_colorbar(figure=figure,
                             ax=colorbar_ax,
                             smfs=smfs,
                             vmin=vmin, vmax=vmax,
                             multicollage=True,
                             colorbar_params=colorbar_params,
                             threshold=threshold,
                             cmap=cmap,
                             **kwargs)
