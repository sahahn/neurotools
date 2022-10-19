from nilearn.connectome.connectivity_matrices import vec_to_sym_matrix
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import nibabel as nib
from ..transform.rois import project_map_fis
from .plot import plot

from matplotlib.colors import Normalize
import seaborn as sns
from .funcs import _get_colors

from textwrap import wrap

def _scale_weights(weights, ref_weights):
    
    # If includes negative weights, we want to scale
    # in a way that preserves positive and negative ... 
    # But we also don't want to assume that the vmax is the same as the vmin.
    # Solution here is to normalize the absolute weights, then re-cast to original sign
    # If all positive to begin with, no harm done
    signs = np.sign(weights)
    ref_signs = np.sign(ref_weights)

    # Fit min max scaler on ref weights, then apply to base weights
    # so scaled weights between 0 and 1, based on all edges
    scaler = MinMaxScaler().fit(np.abs(ref_weights).reshape((-1, 1)))
    scaled_weights = scaler.transform(np.abs(weights).reshape((-1, 1))).reshape(weights.shape)
    ref_scaled_weights = scaler.transform(np.abs(ref_weights).reshape((-1, 1))).reshape(ref_weights.shape)

    # Re-apply signs
    scaled_weights *= signs
    ref_scaled_weights *= ref_signs

    return scaled_weights, ref_scaled_weights

def add_subgraph_plot(sub_G, G, ax, edge_scale=5,
                      layout='spring', x_scale=.8,
                      y_scale=1, color_edges=True,
                      min_node_sz=200, max_node_sz=400):
    
    # Use weights, + ref for edge thickness
    weights = np.array(list(nx.get_edge_attributes(sub_G, 'weight').values()))
    ref_weights = np.array(list(nx.get_edge_attributes(G, 'weight').values()))

    # Get scaled 0-1 (or -1, 1) weights + ref scaled weight w/ signs preserved
    scaled_weights, ref_scaled_weights = _scale_weights(weights, ref_weights)

    # Optionally color edges
    if color_edges:
        colors = _get_colors(scaled_weights, ref_scaled_weights)
        node_color = 'grey'
    else:
        colors = None
        node_color = None

    # Multiply further by edge, after weights potentially used by color
    scaled_weights = scaled_weights * edge_scale

    # Set pos by layout choice
    if layout == 'spring':
        pos = nx.spring_layout(sub_G)
    elif layout == 'circular':
        pos = nx.circular_layout(sub_G)

    # Custom
    else:
        pos = layout(sub_G)

    # Size nodes just by their weighted degree in the full graph
    # Make sure to take absolute - Should abs be before degree?
    # Or maybe it is better to just correlate to the listed weighted degree
    ref_degrees = nx.degree(G, weight='weight')
    degs = [np.abs(ref_degrees[node]) for node in sub_G.nodes()]
    ref_degs = np.abs([x[1] for x in list(ref_degrees)])

    # Init normalize object on the reference degree dist from full G
    # Then apply to just the degrees being plotted in this sub-graph,
    # as further sized by passed constraints
    norm = Normalize(np.min(ref_degs), np.max(ref_degs))
    sizes = (norm(degs).data * (max_node_sz - min_node_sz)) + min_node_sz

    # Plot
    nx.draw(sub_G, pos,
            node_color=node_color,
            node_size=sizes,
            edge_color=colors,
            width=scaled_weights,
            with_labels=True, ax=ax)

    # Scale based on passed params to fix out of bounds labels
    ax.set_xlim([x_scale*x for x in ax.get_xlim()])
    ax.set_ylim([y_scale*y for y in ax.get_ylim()])

def _get_diag(x):
    
    around = int(np.sqrt((x * 2)))
    
    for d_len in range(max([around, around - 5]), around+5):
        
        if x == int(((d_len * d_len) - d_len) / 2):
            return np.zeros(d_len)
        elif x == (d_len * d_len) / 2:
            return None
        
def to_sym(betas):
    '''Assumes betas is numpy array'''
    
    return vec_to_sym_matrix(betas, diagonal=_get_diag(len(betas)))

def _beta_to_G(betas, label_names, n_std=None):
    
    if n_std is None:
        thresh = np.min(betas) - 1
    else:
        thresh = np.mean(betas) + (np.std(betas) * n_std)

    # Make graph
    G = nx.Graph()
    for i in range(len(betas)):
        for j in range(len(betas[i])):
            if betas[i][j] > thresh:
                G.add_edge(label_names[i], label_names[j], weight=betas[i][j])
                
    return G

def _line(fig, gs, i):
    
    line_ax = fig.add_subplot(gs[i, :])
    line_ax.axhline(y=1, ls='--', color='gray')
    line_ax.set_axis_off()
    line_ax.patch.set_alpha(0)

def _plot_brain(data, fig, ax, ref, label_names,
                show_abs=True, is_top=False, colorbar=True,
                threshold='auto', **plot_kwargs):

    # For now, if ref is None, let's assume
    # we are plotting just auto roi names to surface
    # TODO will need to ass more cases
    if ref is None:
        if is_top:
            b_sz = .25
            wspace = 0
        else:
            b_sz = .1
            wspace = .1

        # Call plot
        plot(data,
             view='flat standard', b_sz=b_sz, wspace=wspace,
             figure=fig, ax=ax, colorbar=colorbar, aspect=10,
             shrink=.75, threshold=threshold, **plot_kwargs)

    # 4D prob. case
    elif isinstance(ref, nib.Nifti1Image) and len(ref.shape) == 4:

        if is_top:
            display_mode = 'lyrz'
        else:
            display_mode = 'ortho'

        # Project onto data
        degree_proj = project_map_fis(data, ref, label_names)

        # Then plot in axes
        plot(degree_proj, display_mode=display_mode, axes=ax,
             plot_abs=show_abs, threshold=threshold,
             colorbar=colorbar, **plot_kwargs)

def _add_top_figure(degrees, gs, fig, ref,
                    label_names, show_abs=True, **plot_kwargs):

    # For very top row, / the full degree projection
    # Use sub gridspec for more spacing control
    sub_gs = GridSpecFromSubplotSpec(nrows=1, ncols=3,
                                     subplot_spec=gs[0, :],
                                     width_ratios=[.2,  1, .2],)
    full_ax = fig.add_subplot(sub_gs[1])

    # Make brain plot
    _plot_brain(degrees, fig, full_ax, ref, label_names,
                show_abs=show_abs, is_top=True, colorbar=True, **plot_kwargs)

    # Add title + line
    full_ax.set_title('Projection of All Weighted Degrees', size=24, y=1.05)
    _line(fig, gs, 1)

def _plot_ref(ref, node, top_feats, gs, i,
              label_names, fig, show_abs, **plot_kwargs):
    
    # Gen ref values for plotting roi
    if ref is None:

        # Handle pos / neg only case
        if top_feats[node] < 0:
            add = 1
        else:
            add = -1

        roi_vals = {k: top_feats[node] + add for k in label_names}
        roi_vals[node] = top_feats[node]
        roi_threshold = top_feats[node] + (add / 2)

    else:
        roi_vals = {node: top_feats[node]}
        roi_threshold = None

    # Cast to series
    roi_vals = pd.Series(roi_vals)
    
    # TODO this should be based off plotting vol or surf
    y_adj = 0
    if ref is None:
        y_adj = .05

    # Add reference plot
    ref_ax = fig.add_subplot(gs[i*2, 0])
    ref_ax.text(0, 1, f'{i}). {node}', size=24)
    ref_ax.text(.5, y_adj, f'Weighted Degree: {top_feats[node]:.4f}',
                verticalalignment='center',
                horizontalalignment='center',
                fontstyle='italic')

    # W/ brain
    _plot_brain(roi_vals, fig, ref_ax, ref, label_names,
                show_abs=show_abs, is_top=False, colorbar=False,
                threshold=roi_threshold, **plot_kwargs)
    ref_ax.patch.set_alpha(0)

def _add_row(fig, gs, i, top_feats, ref,
             label_names, show_abs, G, G_norm,
             sub_sz, graph_x_scale=3, graph_y_scale=1, **plot_kwargs):
    
    # Get info needed for passed i
    node = top_feats.index[i-1]

    # Plot reference brain
    _plot_ref(ref=ref, node=node, top_feats=top_feats,
              gs=gs, i=i,
              label_names=label_names, fig=fig,
              show_abs=show_abs, **plot_kwargs)

    # Find x closest nodes to plot sub-graph for sub graph
    path_lens = nx.shortest_path_length(G_norm, source=node, weight='weight')
    sub_nodes = pd.Series(path_lens).sort_values()[:sub_sz].index
    
    # Make plot w/ base G, not G norm
    sub_G = nx.subgraph(G, sub_nodes)
    
    # Add so it goes up into the line ax
    net_ax = fig.add_subplot(gs[(i*2)-1:(i*2)+1, 1])
    net_ax.set_title(f'Subgraph of closest {sub_sz} nodes',
                     size=12, y=-.075, fontstyle='italic')
    add_subgraph_plot(sub_G, G, net_ax, edge_scale=5,
                      layout='circular', x_scale=graph_x_scale,
                      y_scale=graph_y_scale)
    net_ax.patch.set_alpha(0)

    # Get series of all edges from current node - use G
    edges = G.edges(node, data=True)
    edges_vals = pd.Series({e[1] : e[2]['weight'] for e in edges})

    # Add edge's projection plot
    edge_ax = fig.add_subplot(gs[i*2, 2])
    edge_ax.set_title(f'Projection of all edges from {node}', size=14)
    
    _plot_brain(edges_vals, fig, edge_ax, ref, label_names,
                show_abs=show_abs, is_top=False, colorbar=True, **plot_kwargs)
    edge_ax.patch.set_alpha(0)

def _add_bars(degrees, fig, gs, feat_lim=20):

    # Get degree df
    top_feats = degrees.abs().sort_values()[::-1][:feat_lim].index
    df = degrees[top_feats].to_frame().reset_index()
    df = df.rename({'index': 'Node', 0: 'Weighted Degree'}, axis=1)

    # Use full degrees as reference for getting colors
    weights = np.array(df['Weighted Degree'])
    ref_weights = np.array(degrees)
    colors = _get_colors(weights, ref_weights)
    palette = {w: c for w, c in zip(df['Node'], colors)}

    # Plot
    end_ax = fig.add_subplot(gs[-1, :])
    sns.barplot(y='Weighted Degree', x='Node', orient='v', data=df, ax=end_ax, palette=palette)
    end_ax.tick_params(axis='x', labelrotation=87)
    sns.despine(ax=end_ax, top=True, left=True, bottom=True, right=True)
    end_ax.patch.set_alpha(0)

    # Add line at 0
    if np.min(weights) < 0:
        end_ax.axhline(0, color='k', lw=.2)

    # Add grid lines
    end_ax.grid(axis='y', linewidth=1, alpha=.33)

def _get_norm_betas(betas):

    # Abs, and prepare for input
    abs_flat_betas = np.abs(betas).flatten().reshape((-1, 1))

    # Scale to 0-1
    scaled_betas = MinMaxScaler().fit_transform(abs_flat_betas).reshape(betas.shape)

    # Returned normed, so lower weight is higher
    return 1 - scaled_betas

def show_network_fis(fis, label_names, ref=None,
                     top_n=3, sub_sz=5, show_abs=True,
                     feat_lim=20,
                     graph_x_scale=2.75, graph_y_scale=1,
                     text_wrap=35, width=26,
                     v_scale=4.25, **plot_kwargs):

    # Optionally apply text-wrap to long labels
    if label_names is not None and text_wrap is not None:
        label_names = np.array(['-\n'.join(wrap(x, text_wrap)) for x in label_names])

    # Process beta's - assume that we need mean here... (for now)
    betas = to_sym(np.array(fis.mean()))

    if show_abs:
        betas = np.abs(betas)

    # Gen networks + degrees
    G = _beta_to_G(betas, label_names, n_std=None)
    G_norm = _beta_to_G(_get_norm_betas(betas), label_names, n_std=None)

    # Base top feats off top abs degrees, if already abs, then nothing changes
    degrees = pd.Series(dict(nx.degree(G, weight='weight'))).sort_values()
    top_feats = degrees.abs().sort_values()[::-1][:top_n].index

    # Init figure and main Grid Spec
    fig = plt.figure(figsize=(width, v_scale * (top_n+2)))
    
    # The base sz of the dotted line in the plot
    # where size is relative to 1.
    line_sz = .1

    # Init with top figure
    height_ratios = [1]
    for feat in top_feats:

        # Make adjustments for text wrap
        line_adj_sz = line_sz + (feat.count('\n') * line_sz * 2)
        height_ratios += [line_adj_sz, 1]

    # Add last line + bar chart
    height_ratios += [line_sz, 2]

    gs = GridSpec(nrows=(top_n*2)+3, ncols=3, figure=fig,
                  width_ratios=[1.25,  1.25, 1.75],
                  height_ratios=height_ratios,
                  hspace=.15, wspace=-.05)

    # Add top row - main degree plot
    _add_top_figure(degrees, gs=gs, fig=fig,
                    ref=ref, label_names=label_names,
                    show_abs=show_abs, **plot_kwargs)

    # Fill in rest of rows
    for i in range(1, top_n+1):

        _add_row(fig, gs, i, degrees.loc[top_feats], ref,
                label_names, show_abs,
                G=G, G_norm=G_norm, 
                sub_sz=sub_sz,
                graph_x_scale=graph_x_scale,
                graph_y_scale=graph_y_scale,
                **plot_kwargs)
        
        _line(fig, gs, (i*2)+1)

    # Add seaborn bar chart importances
    _add_bars(degrees, fig, gs, feat_lim=feat_lim)




