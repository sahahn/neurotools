import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.lib.shape_base import tile
from .plot_single_surf import plot_single_surf, add_collage_colorbar
from nilearn.plotting import plot_glass_brain, plot_stat_map, plot_roi
from .ref import SurfRef
from ..transform.space import process_space

# Threshold auto fast_abs_percentile(data) - 1e-5 from .._utils.extmath import fast_abs_percentile

def _proc_vs(data, vmin, vmax, symmetric_cbar):

    if vmin is None and vmax is None:
        vmin = np.nanmin([np.nanmin(d) for d in data])
        vmax = np.nanmax([np.nanmax(d) for d in data])

        if np.abs(vmin) > vmax:
            vmax = np.abs(vmin)

            if symmetric_cbar:
                vmin = -vmax
        
        elif symmetric_cbar:
            vmin = -vmax
            
    if vmin is None:
        vmin = np.nanmin([np.nanmin(d) for d in data])
    if vmax is None:
        vmax = np.nanmax([np.nanmax(d) for d in data])

    return vmin, vmax

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
                   symmetric_cbar=False,
                   vmin=None, vmax=None,
                    **kwargs):
    
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
                      figsize=(15, 10), symmetric_cbar=False,
                      figure=None, axes=None, subplot_spec=None,
                      wspace=-.35, hspace=-.1,
                      colorbar=False,
                      dist=6.5, colorbar_params={}, **kwargs):
    '''
    data should be list with two elements [lh, rh] data.

    view: 'standard' means plot left and right hemisphere lateral
    and medial views in a grid. 
    'fb' / front back, means plot anterior posterior views in a row.
    
    If axes is passed, it should be a flat list with 4 axes,
    and if colorbar is True, then the 5th axes passed should be
    the spot for the color bar.
    '''
    
    vmin, vmax = _proc_vs(data, vmin, vmax, symmetric_cbar)
    smfs = []

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
                          threshold=None,
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
                          surf_params={}, vol_params={}):


    # Right now, basically have it be that you cant plot the subcort and surf with
    # dif colorbars, but could have this be more flexible later...
    # Prep vmin, vmax
    vmin, vmax = _proc_vs([surf[0], surf[1], vol.get_fdata()], vmin, vmax, symmetric_cbar)

    # Init smfs with min and max of vol data, as these values won't change in
    # in the same way plotting surf vals can.
    smfs = [np.array([np.min(vol.get_fdata()), np.max(vol.get_fdata())])]

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

    smfs += smfz

    # Plot volume
    vol_ax = figure.add_subplot(vol_grid)

    if vol_plot_type == 'glass':
        vol_plot_func = plot_glass_brain
    elif vol_plot_type == 'stat':
        vol_plot_func = plot_stat_map
    elif vol_plot_type == 'roi':
        vol_plot_func = plot_roi

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

def _proc_defaults(ref, surf_mesh, bg_map, darkness):

    # Allow any user passed args to override space ref defaults here
    if surf_mesh is None:
        surf_mesh = ref.surf_mesh
    if bg_map is None:
        bg_map = ref.bg_map
    if darkness is None:
        darkness = ref.darkness

    return surf_mesh, bg_map, darkness

def _load_data_and_ref(data, space=None, hemi=None):
    
    # Process the data + space - returns data as dict
    data, space = process_space(data, space, hemi)
    
    # If no space, i.e., just sub, return no Ref
    if space is None:
        return data, None

    # Otherwise generate SurfRef with defaults
    ref = SurfRef(space=space)
    
    # fs_LR space defaults
    if 'fs_LR' in space:
        ref.surf_mesh = 'very_inflated'
        ref.bg_map = 'sulc_conte'
        ref.darkness = .5
    
    # Freesurfer spaces defaults
    else:
        ref.surf_mesh = 'inflated'
        ref.bg_map = 'sulc'
        ref.darkness = 1

    return data, ref

def _plot_surf_parc(data, space=None, hemi=None, surf_mesh=None, bg_map=None,
                    cmap='prism', bg_on_data=True, darkness=None,
                    wspace=-.35, hspace=-.1, alpha=1,
                    threshold=.1, colorbar=False, **kwargs):

    # Fine passing already proc'ed whatever here
    data, ref = _load_data_and_ref(data, space=space, hemi=hemi)
    
    # If user-passed
    surf_mesh, bg_map, darkness = _proc_defaults(ref, surf_mesh, bg_map, darkness)
    
    # Both hemi's passed case.
    if 'lh' in data and 'rh' in data:
        plot_surf_collage(data=[data['lh'], data['rh']],
                          ref=ref,
                          surf_mesh=surf_mesh,
                          bg_map=bg_map,
                          cmap=cmap,
                          avg_method='median',
                          threshold=threshold,
                          symmetric_cbar=False,
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
    plot_surf_hemi(data=hemi_data, ref=ref, hemi=hemi,
                   surf_mesh=surf_mesh, bg_map=bg_map,
                   cmap=cmap, avg_method='median',
                   threshold=threshold,
                   symmetric_cbar=False,
                   alpha=alpha,
                   bg_on_data=bg_on_data,
                   darkness=darkness,
                   colorbar=colorbar,
                   **kwargs)

    return

def _plot_surf(data, space=None, hemi=None, surf_mesh=None, bg_map=None,
               cmap='cold_hot', bg_on_data=True, darkness=None,
               avg_method='mean', wspace=-.35, hspace=-.1, alpha=1,
               symmetric_cbar=False, threshold=None,
               colorbar=True, **kwargs):
    
    # Fine passing already proc'ed whatever here
    data, ref = _load_data_and_ref(data, space=space, hemi=hemi)
    
    # If user-passed
    surf_mesh, bg_map, darkness = _proc_defaults(ref, surf_mesh, bg_map, darkness)

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

    surf_args = ['vol_alpha', 'bg_map', 'surf_mash',
                 'darkness']
    surf_params = {key: kwargs[key] for key in surf_args if key in kwargs}
    
    # Special cases
    if 'surf_alpha' in kwargs:
        surf_params['alpha'] = kwargs['surf_alpha']
    if 'surf_hspace' in kwargs:
        surf_params['hspace'] = kwargs['surf_hspace']
    if 'surf_wspace' in kwargs:
        surf_params['wspace'] = kwargs['surf_wspace']

    vol_args = ['resampling_interpolation', 'plot_abs', 'black_bg',
                'display_mode', 'annotate', 'cut_coords',
                'bg_img', 'draw_cross',
                'dim', 'view_type', 'linewidths']
    vol_params = {key: kwargs[key] for key in vol_args if key in kwargs}
    
    # Special case
    if 'vol_alpha' in kwargs:
        vol_params['alpha'] = kwargs['vol_alpha']

    colorbar_args = ['fraction', 'shrink', 'aspect',
                     'pad', 'anchor', 'format', 'cbar_fig_ratio']
    colorbar_params = {key: kwargs[key] for key in colorbar_args if key in kwargs}

    return surf_params, vol_params, colorbar_params
 

def _plot_surf_vol_parc(data, space=None, hemi=None,
                        surf_mesh=None, bg_map=None,
                        colorbar=False, symmetric_cbar=False,
                        darkness=None, vmin=None, vmax=None,
                        cbar_vmin=None, cbar_vmax=None,
                        figure=None, axes=None, subplot_spec=None,
                        figsize=(20, 20), title=None, title_sz=18,
                        hspace=0, wspace=0, surf_alpha=1,
                        threshold=.1, surf_to_vol_ratio=1, **kwargs):

    # Fine passing already proc'ed whatever here
    data, ref = _load_data_and_ref(data, space=space, hemi=hemi)
    
    # If user-passed
    surf_mesh, bg_map, darkness = _proc_defaults(ref, surf_mesh, bg_map, darkness)
    
    # Sort kwargs
    surf_params, vol_params, colorbar_params = _sort_kwargs(kwargs)

    # Add surface specific
    surf_params['surf_mesh'] = surf_mesh
    surf_params['bg_map'] = bg_map
    surf_params['darkness'] = darkness
    surf_params['surf_alpha'] = surf_alpha
    surf_params['ref'] = ref

    plot_surf_vol_collage(surf=[data['lh'], data['rh']],
                          vol=data['sub'],
                          vol_plot_type='roi',
                          cmap='prism',
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
                          surf_to_vol_ratio=surf_to_vol_ratio)

def _plot_surf_vol(data, space=None, hemi=None, **kwargs):
    pass


def plot(data, space=None, hemi=None, **kwargs):
    '''The most automated magic plotting function avaliable,
    used to plot a wide range of neuroimaging data (volumes / surfs).

    Parameters
    -----------
    data : loc or data array
        The data in which to plot, either statistical map or parcellaction,
        as broadly either a single surface,
        collage of surfaces, a single
        volume, or collage with surfaces and volume.
    
        Data can be passed in many ways:

        1. A single data representing surf, surf+surf, surf+surf+sub or just sub
        2. A single file location w/ a data array
        3. A list / array-like of length either 2 or 3, if 2 then represents surf+surf
        if 3 then surf+surf+sub
        4. A list / array-like same as above, but with file-paths

    space : None or str, optional
        If left as default, the space in which
        the data to plot is in will be automatically inferred,
        otherwise this parameter can be manually set,
        overriding the automatically detected choice.

        Current supported options are:

        - 'fsaverage'
        - 'fsaverage5'
        - '32k_fs_LR'
        - '164k_fs_LR'

        ::

            default = None

    hemi : None or str, optional
        If left as default, then this option will
        be automatically set (with if only one hemisphere of
        data passed defaulting to left hemisphere).

        Otherwise, you may override the automatically
        detected hemisphere by passing either just 'lh' or 'rh'
        when plotting only a single hemisphere's data.

        ::

            default = None


    kwargs : kwargs style arguments
        There are number of different plotting specific arguments
        that can optionally be modified. The default values for these arguments
        may change also depending on different auto-detected settings, i.e.,
        if plotting a single surface vs. surface collage, or if plotting
        a volumetric image.

        - surf_mesh : A str indicator specifying a valid surface mesh,
            with respect to the current space,
            in which to plot the data on. The default for
            freesurfer spaces is 'inflated' and for fs_LR spaces
            is 'very_inflated'.

        - bg_map : A str indicator specifying a valid array of values,
            with respect to the current space, in which to use as a
            background map when plotting. This map is plotted in greyscale
            underneath the data points, often used for realistic shading.
            The default for freesurfer spaces is 'sulc' and for fs_LR spaces
            is 'sulc_conte'.

        - cmap : An instance of :class:`matplotlib.colors.Colormap` or str
            representing the name of a matplotlib colormap. This will be the color map
            in which the values are plotted according to. When plotting
            surface parcellations the default cmap is 'prism', when
            plotting statistical maps, the default cmap is 'col_hot'.

        - bg_on_data : temp

        - darkness : temp

        - avg_method : temp

        - vol_plot_type : {'glass', 'stat', 'roi'}

        - alpha : temp

        - colorbar : temp

        - symmetric_cbar : temp

        - threshold : temp

        - wspace : temp

        - hspace : temp

    

    For example, if plotting a multi-figure surface + volume collage,
    then you can take full control on placement and size of sub-figures via the following parameters:
    
    - hspace
    - wspace
    - surf_hspace
    - surf_wspace
    - surf_to_vol_ratio
    - cbar_fig_ratio

    '''
    
    # Load / perform initial auto-detection of data
    data, _ = _load_data_and_ref(data, space=space, hemi=hemi)

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
            _plot_surf_parc(data, space=space, hemi=hemi, **kwargs)
        
        # Just volume case
        elif 'lh' not in data and 'rh' not in data:

            # Override vol_plot_type - force roi
            plot_volume(data['sub'], vol_plot_type='roi', **kwargs)
        
        # Last case is vol / surf collage
        else:
            _plot_surf_vol_parc(data, space=None, hemi=None, **kwargs)

    
    # Otherwise - plot stat values
    else:

        # Just surface or surfaces case
        if 'sub' not in data:
            _plot_surf(data, space=space, hemi=hemi, **kwargs)

        # Just volume case
        elif 'lh' not in data and 'rh' not in data:
            plot_volume(data['sub'], **kwargs)

        # Last case is vol / surf collage
        else:
            _plot_surf_vol_parc(data, space=None, hemi=None, **kwargs)


def plot_volume(vol, vol_plot_type='glass', cmap=None,
                threshold=None, vmax=None,
                figure=None, axes=None, **kwargs):

    if vol_plot_type == 'glass':
        vol_plot_func = plot_glass_brain
    elif vol_plot_type == 'stat':
        vol_plot_func = plot_stat_map
    elif vol_plot_type == 'roi':
        vol_plot_func = plot_roi

    vol_plot_func(vol,
                  figure=figure, axes=axes, 
                  cmap=cmap, threshold=threshold, vmax=vmax,
                  **kwargs)