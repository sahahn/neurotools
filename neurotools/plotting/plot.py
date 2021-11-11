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
        
        # Unpack if nifti image
        if isinstance(item, nib.Nifti1Image):
            item = item.get_fdata()

        # Add to list - as make sure array and flat
        collapsed.append(np.array(item).flatten())
    
    # Return as concat version
    return np.concatenate(collapsed)


def _get_if_sym_cbar(data, symmetric_cbar, rois=False):
    '''Assumes data is in standard {} form.'''
     
    # If user passed, keep that value
    if symmetric_cbar != 'auto':
        return symmetric_cbar
    
    # If rois, default = False
    if rois:
        return False
    
    # Get data as 1D array
    flat_data = _collapse_data(data)

    # If all positive or negative, assume false
    if np.all(flat_data >= 0) or np.all(flat_data <= 0):
        return False

    # Otherwise, assume is symmetric
    return True


def _setup_fig_axes(figure, axes,
                    subplot_spec,
                    get_grid, figsize,
                    n_rows, n_cols, widths,
                    heights, proj_3d,
                    title, title_sz,
                    colorbar, colorbar_params,
                    wspace, hspace):

    colorbar_ax = None

    # If no axes or figure is passed,
    if axes is None:
        if figure is None:
            figure = plt.figure(figsize=figsize)
        
        if colorbar is True:

            if 'cbar_fig_ratio' in colorbar_params:
                widths += [colorbar_params['cbar_fig_ratio']]
            else:
                widths += [.5]

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
        
        if colorbar:

            if n_rows == 1:
                colorbar_ax = figure.add_subplot(grid[-1])
            else:
                colorbar_ax = figure.add_subplot(grid[:, -1])
            
            colorbar_ax.set_axis_off()

        if title is not None:

            if colorbar:
                if n_rows == 1:
                    title_ax = figure.add_subplot(grid[:-1])
                else:
                    title_ax = figure.add_subplot(grid[:,:-1])

            else:
                title_ax = figure.add_subplot(grid[:])
            
            title_ax.set_title(title, fontsize=title_sz)
            title_ax.set_axis_off()

        if get_grid:
            return figure, grid, colorbar_ax
        
        else:
            axes = []
            for i in range(n_rows):
                for j in range(n_cols):

                    if n_rows == 1:
                        axes.append(figure.add_subplot(grid[j], projection=proj_3d[i][j]))
                    elif len(widths) == 1:
                        axes.append(figure.add_subplot(grid[i], projection=proj_3d[i][j]))
                    else:
                        axes.append(figure.add_subplot(grid[i, j], projection=proj_3d[i][j]))

            return figure, axes, colorbar_ax

    # If axes passed, but no figure passed
    if figure is None:
        figure = axes.get_figure()

    # If passed axes explicitly, then doesn't matter what the get_grid param is
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
                      title=None, title_sz=18,
                      vmin=None, vmax=None,
                      cbar_vmin=None, cbar_vmax=None,
                      figsize=(15, 10), symmetric_cbar='auto',
                      figure=None, axes=None, subplot_spec=None,
                      wspace=-.35, hspace=-.1,
                      colorbar=False, dist=6.5,
                      colorbar_params={}, **kwargs):
    '''
    data should be list with two elements [lh, rh] data.

    view: 'standard' means plot left and right hemisphere lateral
    and medial views in a grid. 
    'fb' / front back, means plot anterior posterior views in a row.
    
    If axes is passed, it should be a flat list with 4 axes,
    and if colorbar is True, then the 5th axes passed should be
    the spot for the color bar.
    '''

    # Proc if sym cbar auto - assume rois False here
    symmetric_cbar = _get_if_sym_cbar(data, symmetric_cbar, rois=False)
    
    # Get vmin and vmax based off data + passed args
    vmin, vmax = _proc_vs(data, vmin, vmax, symmetric_cbar)
    smfs = []
    
    # Set params based on requested view
    if view in ['standard', 'default']:
        n_rows, n_cols = 2, 2
        hemis = ['lh', 'rh', 'lh', 'rh']
        views = ['lateral', 'lateral', 'medial', 'medial']
        widths = [1, 1]

    elif view in ['fb', 'front back', 'ap', 'anterior posterior']:
        n_rows, n_cols = 1, 5
        hemis = ['lh', 'rh', 'b', 'lh', 'rh']
        views = ['anterior', 'anterior', 'b',
                 'posterior', 'posterior']
        widths = [1, 1, .5, 1, 1]

    elif view in ['f', 'front', 'a', 'anterior']:
        n_rows, n_cols = 1, 2
        hemis = ['lh', 'rh']
        views = ['anterior', 'anterior']
        widths = [1, 1]

    elif view in ['b', 'back', 'p', 'posterior']:
        n_rows, n_cols = 1, 2
        hemis = ['lh', 'rh']
        views = ['posterior', 'posterior']
        widths = [1, 1]

    proj_3d = [['3d' for _ in range(n_cols)] for _ in range(n_rows)]

    # Setup figres and axes
    figure, axes, colorbar_ax =\
        _setup_fig_axes(figure, axes, subplot_spec, False,
                        figsize, n_rows, n_cols,
                        widths, None, proj_3d, title, title_sz,
                        colorbar, colorbar_params,
                        wspace, hspace)
        
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
                          figure=None, axes=None,
                          subplot_spec=None,
                          figsize=(20, 20),
                          title=None, title_sz=18,
                          colorbar=False,
                          symmetric_cbar=False,
                          hspace=0, wspace=0,
                          colorbar_params={},
                          surf_params={}, vol_params={},
                          verbose=0, _print=None):

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
    n_rows, n_cols = 2, 1
    widths = [1]
    heights = [surf_to_vol_ratio] + [1]
    proj_3d = [['3d'], [None]]

    # Setup figures and axes
    figure, grid, colorbar_ax =\
        _setup_fig_axes(figure, axes, subplot_spec, True,
                        figsize, n_rows, n_cols,
                        widths, heights, proj_3d, title, title_sz,
                        colorbar, colorbar_params,
                        wspace, hspace)
    
    # Either multi-index w/ colorbar or just two rows w/o
    if colorbar:
        surf_grid, vol_grid = grid[0, 0], grid[1, 0]
    else:
        surf_grid, vol_grid = grid[0], grid[1]

    # Plot surf collage in the top grid spot
    figure, axes, smfz = plot_surf_collage(surf, 
                                           figure=figure,
                                           axes=None,
                                           subplot_spec=surf_grid,
                                           cmap=cmap,
                                           threshold=threshold,
                                           vmin=vmin,
                                           vmax=vmax,
                                           colorbar=False,
                                           **surf_params)
    
    # Keep track of vals for plotting colorbar
    smfs += smfz

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

    return figure, smfs

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
        ref.bg_map = 'sulc_conte'
        ref.darkness = .5
    
    # Freesurfer spaces defaults
    else:
        ref.surf_mesh = 'inflated'
        ref.bg_map = 'sulc'
        ref.darkness = 1

    return data, ref

def _proc_avg_method(data, rois, avg_method):

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

def _prep_auto_defaults(data, space, hemi, rois,
                        symmetric_cbar, threshold,
                        cmap, colorbar, avg_method, _print):

    # Fine passing already proc'ed whatever here
    data, ref = _load_data_and_ref(data, space=space, hemi=hemi, _print=_print)

    # A bunch of funcs use collapse data, so quicker to just
    # do it once here and pass that to funcs
    flat_data = _collapse_data(data)

    # Process automatic symmetric_cbar
    symmetric_cbar = _get_if_sym_cbar(flat_data, symmetric_cbar, rois=rois)

    # Proc threshold if auto
    threshold = _proc_threshold(flat_data, threshold, rois=rois)

    # Get cmap based on passed + if rois or not + if sym
    cmap = _proc_cmap(cmap, rois, symmetric_cbar)
    
    # Proc colorbar option - yes if not rois
    if colorbar == 'default':
        colorbar = True
        if rois:
            colorbar = False
    
    # Proc avg method default
    avg_method= _proc_avg_method(data, rois, avg_method)

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
        plot_surf_collage(data=[data['lh'], data['rh']],
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
        return

    # Just lh/ rh cases
    if 'lh' in data:
        hemi_data, hemi = data['lh'], 'left'
    if 'rh' in data:
        hemi_data, hemi = data['rh'], 'right'
    
    # Plot
    plot_surf_hemi(data=hemi_data,
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

    return

def _sort_kwargs(kwargs):

    # Surface specific
    surf_args = ['vol_alpha', 'bg_map',
                 'surf_mash', 'darkness', 'view', 'dist']
    surf_params = {key: kwargs[key] for key in surf_args if key in kwargs}
    
    # Special cases
    if 'surf_hspace' in kwargs:
        surf_params['hspace'] = kwargs['surf_hspace']

    # If also passed directly
    if 'surf_params' in kwargs:
        surf_params = {**kwargs['surf_params'], **surf_params}
    
    # Volume specific
    vol_args = ['resampling_interpolation', 'plot_abs',
                'black_bg', 'display_mode', 'annotate', 'cut_coords',
                'bg_img', 'draw_cross',
                'dim', 'view_type', 'linewidths']
    vol_params = {key: kwargs[key] for key in vol_args if key in kwargs}

    # Special case
    if 'vol_alpha' in kwargs:
        vol_params['alpha'] = kwargs['vol_alpha']

    # If also passed directly
    if 'vol_params' in kwargs:
        vol_params = {**kwargs['vol_params'], **vol_params}
    
    # Colorbar specific
    colorbar_args = ['fraction', 'shrink', 'aspect',
                     'pad', 'anchor', 'format', 'cbar_fig_ratio']
    colorbar_params = {key: kwargs[key] for key in colorbar_args if key in kwargs}

    # If also passed directly
    if 'colorbar_params' in kwargs:
        colorbar_params = {**kwargs['colorbar_params'], **colorbar_params}

    return surf_params, vol_params, colorbar_params

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
                   figure=None, axes=None, subplot_spec=None,
                   figsize='default', title=None, title_sz=18,
                   hspace='default', wspace=-.2, surf_alpha=1,
                   avg_method='default', threshold='auto',
                   surf_to_vol_ratio='default', surf_wspace='default',
                   bg_on_data=.25,
                   _print=None, **kwargs):

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
    surf_params['darkness'], surf_params['alpha'] = darkness, surf_alpha
    surf_params['ref'], surf_params['avg_method'] = ref, avg_method
    surf_params['wspace'], surf_params['bg_on_data'] = surf_wspace, bg_on_data
    
    # Pass arguments to plot surf vol collage
    plot_surf_vol_collage(surf=[data['lh'], data['rh']],
                          vol=data['sub'],
                          vol_plot_type=vol_plot_type,
                          cmap=cmap,
                          threshold=threshold,
                          vmin=vmin, vmax=vmax,
                          cbar_vmin=cbar_vmin,
                          cbar_vmax=cbar_vmax,
                          figure=figure, axes=axes,
                          subplot_spec=subplot_spec,
                          figsize=figsize,
                          title=title, title_sz=title_sz,
                          colorbar=colorbar,
                          symmetric_cbar=symmetric_cbar,
                          hspace=hspace, wspace=wspace,
                          colorbar_params=colorbar_params,
                          surf_params=surf_params,
                          vol_params=vol_params,
                          surf_to_vol_ratio=surf_to_vol_ratio,
                          _print=_print)


def plot(data, space=None, hemi=None, verbose=0, **kwargs):
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
          Dictionary keys must be one or more of 'lh', 'rh' and 'sub',
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
          surface data, and the last is subcortical data, all to
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

        - '32k_fs_LR' : HCP standard 32k vertex space. 

        - '59k_fs_LR' : HCP standard 59k vertex space. 

        - '164k_fs_LR' : HCP standard 164k vertex space. Note, while this space has the same number of vertex as fsaverage space, it is not the same.

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

    kwargs : **kwargs
        There are number of different plotting specific arguments
        that can optionally be modified. The default values for these arguments
        may change also depending on different auto-detected settings, i.e.,
        if plotting a single surface vs. surface collage, or if plotting
        a volumetric image.

        - surf_mesh : str or 2D array
            A str indicator specifying a valid surface mesh,
            with respect to the current space, or a two dimensional array
            with information about the coordinates and vertex-faces
            in which to plot the data on. The default for
            freesurfer spaces is 'inflated' and for fs_LR spaces
            is 'very_inflated'. If you wish to plot surface
            data in native subject space, then this argument must be
            supplied with a loaded 2D array / valid mesh, and data
            must be passed in dictionary style.

            This parameter is only used when plotting surfaces.

        - bg_map : str or 2D array
            This argument specifies
            a background map to be used when plotting. This
            should be passed as either a str indicator specifying a valid file
            within the current surface space, or as a valid array of values,
            again, with respect to the current space. This map is plotted in greyscale
            underneath the data points, often used for realistic shading.
            The default for when plotting freesurfer spaces is 'sulc'
            and for fs_LR spaces is 'sulc_conte'.

            This parameter is only used when plotting surfaces.

        - cmap : str or :class:`matplotlib.colors.Colormap`
            This should be pass as an instance
            of :class:`matplotlib.colors.Colormap` or str
            representing the name of a matplotlib colormap. This will be the color map
            in which the values are plotted according to. When plotting surface
            parcellations, the default cmaps are 'prism' if plotting rois, 'Reds'
            if plotting not symmetric statistical maps, and 'cold_hot' if plotting
            symmetric statistical maps (e.g., values above and below 0).

        - vol_plot_type : {'glass', 'stat', 'roi'}
            This parameter control the type of volumetric plot in which to generate,
            with valid values as one of 'glass', 'stat' and 'roi'. By default,
            if detected to be plotting a parcellation, the value 'roi' is used.
            Otherwise, the default volumetric plotting type is 'glass'.
            The corresponding back-end nilearn functions used are:

            - 'glass': :func:`nilearn.plotting.plot_glass_brain`
            - 'stat': :func:`nilearn.plotting.plot_stat_map`
            - 'roi': :func:`nilearn.plotting.plot_roi`

            This parameter is only used when plotting volumetric data.

        - bg_on_data : bool or float
            If True, and a bg_map is specified,
            the data to plot is multiplied by the background image, 
            so that e.g. sulcal depth is visible beneath the surf_data.

            If passed as a float value, then that will trigger multiplying
            the data to plot by the background image times the passed bg_on_data
            value. So for example passing True is equivalent to passing 1 and passing
            False is the same as passing 0.
            This allows for fine tuning how much the background image
            is melded with the data shown. The default value is .25.

            This parameter is only used when plotting surfaces.

        - view : str
            If plotting a single surface hemisphere, this
            parameter must be one of:
            {'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'}
            Where 'lateral' is the default view if none is set.

            If instead plotting a collage of surface views, then
            valid view parameters are one of:
            {'standard', 'front back', 'front', 'back'}
            which correspond to different preset collections of surface views.
            The default is 'standard'.
            
            Note: If plotting a collage of surface and volumetric views,
            it is reccomended to keep the default collage 'standard' view,
            as the other views have not yet been properly formatted yet.

        - figsize : 'default' or (int, int)
            This parameter is used when axes are not passed, and
            a new figure is being generated. It represents
            the size of the underlying matplotlib figure to plot on.
            The default value for this function varies based on what type of
            plot is being made (e.g., not the same for surface collage vs. single surface).

        - darkness : temp

        - avg_method : temp

        - alpha : temp

        - colorbar : temp

        - symmetric_cbar : temp

        - threshold : temp

        - wspace : temp

        - hspace : temp


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
        unique_vals = np.unique(data['sub'].get_fdata())

    # If all of the data points are interger-like, assume
    # we are plotting a parcellation and not stat data
    if all([float(u).is_integer() for u in unique_vals]):
        
        # Just surface or surfaces case
        if 'sub' not in data:
            _plot_surfs(data, space=space, hemi=hemi, rois=True,
                        _print=_print, **kwargs)
        
        # Just volume case
        elif 'lh' not in data and 'rh' not in data:

            # Override vol_plot_type - force roi
            plot_volume(data['sub'], vol_plot_type='roi', **kwargs)
        
        # Last case is vol / surf collage
        else:
            _plot_surfs_vol(data, space=None, hemi=None, rois=True,
                            _print=_print, **kwargs)
    
    # Otherwise - plot stat values
    else:

        # Just surface or surfaces case
        if 'sub' not in data:
            _plot_surfs(data, space=space, hemi=hemi,
                        rois=False, _print=_print, **kwargs)

        # Just volume case
        elif 'lh' not in data and 'rh' not in data:
            plot_volume(data['sub'], **kwargs)

        # Last case is vol / surf collage
        else:
            _plot_surfs_vol(data, space=None, hemi=None,
                            rois=False, _print=_print, **kwargs)


def plot_volume(vol,
                vol_plot_type='glass',
                cmap=None,
                threshold='auto',
                vmax=None,
                figure=None,
                axes=None,
                **kwargs):

    if vol_plot_type == 'glass':
        vol_plot_func = plot_glass_brain
    elif vol_plot_type == 'stat':
        vol_plot_func = plot_stat_map
    elif vol_plot_type == 'roi':
        vol_plot_func = plot_roi

    # Proc threshold if auto
    rois = vol_plot_type == 'roi'
    threshold = _proc_threshold(vol, threshold, rois=rois)
    
    # Call vol plot function
    vol_plot_func(vol,
                  figure=figure,
                  axes=axes, 
                  cmap=cmap,
                  threshold=threshold,
                  vmax=vmax,
                  **kwargs)