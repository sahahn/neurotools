from ..measures import ThresholdNetworkMeasures
import numpy as np

def test_threshold_network_measures_abs():

    nm =\
        ThresholdNetworkMeasures(threshold=0.2,
                                 threshold_type='abs',
                                 threshold_method='value',
                                 to_compute='avg_degree')

    X = np.array([[.1, .2, .3, -.5],
                  [.2, 0, -.5, 0],
                  [.3, -.5, 0, 0],
                  [-.5, 0, 0, 0]])

    # Fit
    nm.fit(X)
    assert nm._feat_names[0] == 'avg_degree'
    assert len(nm._feat_names) == 1

    # Threshold should stay fixed
    nm._threshold_check(X)
    assert nm.threshold == .2

    thresh_X = nm._apply_threshold(X)
    assert len(np.unique(thresh_X)) == 2

    # Should just drop .1 edge
    assert np.sum(thresh_X) == 8

    import networkx as nx
    G = nx.from_numpy_array(thresh_X)

    degrees = [n[1] for n in G.degree()]
    X_trans = nm.transform(X)
    assert len(X_trans) == 1
    assert np.mean(degrees) == X_trans[0]


def test_threshold_network_measures_neg():

    nm =\
        ThresholdNetworkMeasures(threshold=-.2,
                                 threshold_type='neg',
                                 threshold_method='value')

    X = np.array([[.1, .2, .3, -.5],
                  [.2, 0, -.5, 0],
                  [.3, -.5, 0, 0],
                  [-.5, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
    assert np.sum(thresh_X) == 4
    assert thresh_X[0][-1] == 1
    assert thresh_X[0][1] == 0


def test_threshold_network_measures_pos():


    nm =\
        ThresholdNetworkMeasures(threshold=.2,
                                 threshold_type='pos',
                                 threshold_method='value')

    X = np.array([[.1, .2, .3, -.5],
                  [.2, 0, -.5, 0],
                  [.3, -.5, 0, 0],
                  [-.5, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
    assert np.sum(thresh_X) == 4
    assert thresh_X[0][-1] == 0
    assert thresh_X[0][1] == 1


def test_threshold_network_measures_density():

    nm =\
        ThresholdNetworkMeasures(threshold=.2,
                                 threshold_type='abs',
                                 threshold_method='density')

    X = np.array([[0, .2, .3, -.4],
                  [.2, 0, -.5, 0],
                  [.3, -.5, 0, 0],
                  [-.4, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
   
    # 20% of 16 possible edges
    # rounded up is 4
    assert np.sum(thresh_X) == 4

def test_threshold_network_measures_density_pos():

    nm =\
        ThresholdNetworkMeasures(threshold=.2,
                                 threshold_type='pos',
                                 threshold_method='density')

    X = np.array([[0, .2, .3, -.4],
                  [.2, 0, -.5, 0],
                  [0, -.5, 0, 0],
                  [-.4, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
   
    # 20% of 16 possible edges
    # rounded up is 4
    assert np.sum(thresh_X) == 3

def test_threshold_network_measures_density_neg():

    nm =\
        ThresholdNetworkMeasures(threshold=.2,
                                 threshold_type='neg',
                                 threshold_method='density')

    X = np.array([[0, .2, .3, -.4],
                  [.2, 0, -.5, 0],
                  [.3, 0, 0, 0],
                  [-.4, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
   
    # 20% of 16 possible edges
    # rounded up is 4
    assert np.sum(thresh_X) == 3

def test_threshold_network_measures_abs():

    nm =\
        ThresholdNetworkMeasures(threshold=0.2,
                                 threshold_type='abs',
                                 threshold_method='value',
                                 to_compute='avg_degree')

    X = np.array([[.1, .2, .3, -.5],
                  [.2, 0, -.5, 0],
                  [.3, -.5, 0, 0],
                  [-.5, 0, 0, 0]])

    # Fit
    nm.fit(X)
    assert nm._feat_names[0] == 'avg_degree'
    assert len(nm._feat_names) == 1

    # Threshold should stay fixed
    nm._threshold_check(X)
    assert nm.threshold == .2

    thresh_X = nm._apply_threshold(X)
    assert len(np.unique(thresh_X)) == 2

    # Should just drop .1 edge
    assert np.sum(thresh_X) == 8

    import networkx as nx
    G = nx.from_numpy_array(thresh_X)

    degrees = [n[1] for n in G.degree()]
    X_trans = nm.transform(X)
    assert len(X_trans) == 1
    assert np.mean(degrees) == X_trans[0]


def test_threshold_network_measures_neg():

    nm =\
        ThresholdNetworkMeasures(threshold=-.2,
                                 threshold_type='neg',
                                 threshold_method='value')

    X = np.array([[.1, .2, .3, -.5],
                  [.2, 0, -.5, 0],
                  [.3, -.5, 0, 0],
                  [-.5, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
    assert np.sum(thresh_X) == 4
    assert thresh_X[0][-1] == 1
    assert thresh_X[0][1] == 0


def test_threshold_network_measures_pos():


    nm =\
        ThresholdNetworkMeasures(threshold=.2,
                                 threshold_type='pos',
                                 threshold_method='value')

    X = np.array([[.1, .2, .3, -.5],
                  [.2, 0, -.5, 0],
                  [.3, -.5, 0, 0],
                  [-.5, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
    assert np.sum(thresh_X) == 4
    assert thresh_X[0][-1] == 0
    assert thresh_X[0][1] == 1


def test_threshold_network_measures_density():


    nm =\
        ThresholdNetworkMeasures(threshold=.2,
                                 threshold_type='abs',
                                 threshold_method='density')

    X = np.array([[0, .2, .3, -.4],
                  [.2, 0, -.5, 0],
                  [.3, -.5, 0, 0],
                  [-.4, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
   
    # 20% of 16 possible edges
    # rounded up is 4
    assert np.sum(thresh_X) == 4

def test_threshold_network_measures_density_pos():


    nm =\
        ThresholdNetworkMeasures(threshold=.2,
                                 threshold_type='pos',
                                 threshold_method='density')

    X = np.array([[0, .2, .3, -.4],
                  [.2, 0, -.5, 0],
                  [0, -.5, 0, 0],
                  [-.4, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
   
    # 20% of 16 possible edges
    # rounded up is 4
    assert np.sum(thresh_X) == 3

def test_threshold_network_measures_density_neg():


    nm =\
        ThresholdNetworkMeasures(threshold=.2,
                                 threshold_type='neg',
                                 threshold_method='density')

    X = np.array([[0, .2, .3, -.4],
                  [.2, 0, -.5, 0],
                  [.3, 0, 0, 0],
                  [-.4, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
   
    # 20% of 16 possible edges
    # rounded up is 4
    assert np.sum(thresh_X) == 3
