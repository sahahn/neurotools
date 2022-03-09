import numpy as np
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin


def lovain_modularity(G):
    import networkx.algorithms.community as nx_comm

    return nx_comm.modularity(G, nx_comm.louvain_communities(G))

class ThresholdNetworkMeasures(BaseEstimator, TransformerMixin):
    '''This class is designed for thresholding and then extracting network
    measures from an input correlation matrix.

    Parameters
    -----------
    threshold : float, optional
        A floating point threshold between 0 and 1.
        This represents the threshold at which a connection
        in the passed data needs to be in order for it to
        be set as an edge. The type of threshold_method
        also changes how this threshold behaves.

        If 'density', then this value represents the
        percent of edges to keep, out of all possible edges.

        ::

            default = .2

    threshold_type : {'abs', 'pos', 'neg'}
        The type of thresholding, e.g., should the threshold
        be applied to:

        - 'abs'
            The absolute value of the connections

        - 'pos'
            Only consider edges as passed the threshold if >= self.threshold

        - 'neg'
            Only consider edges as passed the if <= self.threshold

        ::

            default = 'abs'

    threshold_method : {'value', 'density'}, optional
        The method for thresholding. The two
        defined options are either to define an edge
        strictly by value, e.g., if threshold_type is 'abs',
        and threshold is .2, then any connection greater than or
        equal to .2 will be set as an edge.

        Alternatively, you may specify that the threshold be
        treated as a density. What this means is that if the threshold
        is set to for example .2, then the top 20% of edges by weight
        will be set as an edge, regardless of the actual value of the edge.

        The passed percentage will be considered
        out of all the possible edges. This will be used to
        select a threshold value, rounding up if needed, then
        all edges above or equal to the threshold will be kept
        (positive or abs case) or in neg case, all edges less than or equal.

        ::

            default = 'value'

    to_compute : valid_measure or list of, optional
        | Either a single str representing a network
            measure to compute, or a list of valid
            measures. You may also pass any custom function
            which accepts one argument G, and returns
            a value.

        | The following global measures are currently implemented
            as options:
        |


        - 'avg_cluster':
            :func:`networkx.algorithms.cluster.average_clustering`
        - 'assortativity':
            :func:`networkx.algorithms.assortativity.degree_assortativity_coefficient`
        - 'global_eff':
            :func:`networkx.algorithms.efficiency_measures.global_efficiency`
        - 'local_eff':
            :func:`networkx.algorithms.efficiency_measures.local_efficiency`
        - 'sigma':
            :func:`networkx.algorithms.smallworld.sigma`
        - 'omega`:
            :func:`networkx.algorithms.smallworld.omega`
        - 'transitivity':
            :func:`networkx.algorithms.cluster.transitivity`

        |
        | You may also select from one of the following
            averages of local measures:
        |

        - 'avg_eigenvector_centrality':
            :func:`networkx.algorithms.centrality.eigenvector_centrality_numpy`
        - 'avg_closeness_centrality':
            :func:`networkx.algorithms.centrality.closeness_centrality`
        - 'avg_degree':
            Average graph degree.
        - 'avg_triangles':
            :func:`networkx.algorithms.cluster.triangles`
        - 'avg_pagerank':
            :func:`networkx.algorithms.link_analysis.pagerank_alg.pagerank`
        - 'avg_betweenness_centrality':
            :func:`networkx.algorithms.centrality.betweenness_centrality`
        - 'avg_information_centrality':
            :func:`networkx.algorithms.centrality.information_centrality`
        - 'avg_shortest_path_length':
            :func:`networkx.algorithms.shortest_paths.generic.average_shortest_path_length`


        ::

            default = 'avg_degree'

    ensure : {None, 'edge', 'is_conn', float}, optional
        When constructing the thresholded network,
        you may optionally specify what trait of the network to
        ensure according to the passed threshold and threshold type.
        By default, this value will be set to 'edge', which means,
        if the passed settings do not ensure atleast 1 edge, the
        threshold will be incremented by the ensure_incr, default = .01,
        until there is atleast one edge.

        Alternatively, you may pass 'is_conn' to specify that
        the network should be connected, that is, every node can be reached by
        every other. Or None, to do
        no checks (which may result in some kind of error).

        Lastly, you may pass a float value between 0 and 1, which
        specifies that the criteria check the size of the largest connected
        component relative to the total number of nodes. For example, passing
        1 is the same as passing 'is_conn', where the largest connected component is
        the same size as the network, and a value of .5 would be that atleast
        half the nodes are connected to each other.

        ::

            default = 'edge'

    ensure_incr : float, optional
        The amount in which to increment each time when checking
        for an ensure criteria. Note if ensure is None, this
        parameter is not used. Also note that this value in the case of threshold_method by value,
        is repeatedly decremented from the original passed
        threshold, unless the threshold type is 'neg', in that case it will be incremented.

        In the case of threshold_type is density, this value will always be incremented,
        as only positive thresholds make sense for density.

        ::

            default = .01

    '''

    def __init__(self, threshold=.2,
                 threshold_type='abs',
                 threshold_method='value',
                 to_compute='avg_degree',
                 ensure='edge',
                 ensure_incr=.01):

        self.threshold = threshold
        self.threshold_type = threshold_type
        self.threshold_method = threshold_method
        self.to_compute = to_compute
        self.ensure = ensure
        self.ensure_incr = ensure_incr

        self._param_checks()

    def _param_checks(self):

        # Input validation
        if self.threshold_method not in ['value', 'density']:
            raise RuntimeError('threshold method must be value or density!')

        # Bad Combo
        if self.threshold_method == 'density' and self.threshold <= 0:
            raise RuntimeError('When using density, the passed threshold must be greater than 0.')

        # Bad params
        if isinstance(self.ensure, float):
            if self.ensure > 1 or self.ensure < 0:
                raise RuntimeError('ensure must be between 0 and 1')

    @property
    def feat_names_(self):
        '''The list of feature names returned
        by this objects transform function. This property
        is special in that it can interact with :class:`BPt.Loader`,
        passing along feature name information.
        '''
        return self._feat_names

    @feat_names_.setter
    def feat_names_(self, feat_names):
        self._feat_names = feat_names

    def fit(self, X=None, y=None):
        '''X is a 2d correlation matrix'''

        # Check here also since this estimator can be
        # used nested in pipeline w/ param searches or
        # outside of ML
        self._param_checks()
        
        # Store passed as initiaL
        self._threshold = self.threshold

        if isinstance(self.to_compute, str):
            self.to_compute = [self.to_compute]

        # The dictionary of valid options
        self._func_dict = {
            'avg_cluster': (nx.average_clustering, False),
            'assortativity': (nx.degree_assortativity_coefficient, False),
            'global_eff': (nx.global_efficiency, False),
            'local_eff': (nx.local_efficiency, False),
            'sigma': (nx.sigma, False),
            'omega': (nx.omega, False),
            'transitivity': (nx.transitivity, False),
            'lovain_modularity' : (lovain_modularity, False),
            'avg_eigenvector_centrality': (nx.eigenvector_centrality_numpy, True),
            'avg_closeness_centrality': (nx.closeness_centrality, True),
            'avg_degree': (self._avg_degree, False),
            'avg_triangles': (nx.triangles, True),
            'avg_pagerank': (nx.pagerank, True),
            'avg_betweenness_centrality': (nx.betweenness_centrality, True),
            'avg_information_centrality': (nx.information_centrality, True),
            'avg_shortest_path_length': (nx.average_shortest_path_length, False),
            'eigenvector_centrality': (nx.eigenvector_centrality_numpy, False),
        }

        return self

    def fit_transform(self, X, y=None):
        '''Fit, then transform a passed 2D numpy correlation matrix.

        Parameters
        ----------
        X : numpy array
            A 2D numpy array representing an input correlation
            matrix.

        Returns
        ---------
        X_trans : numpy array
            Returns a flat array of length number of
            measures in parameter to_compute, representing
            the calculated network statistics.
        '''

        return self.fit(X, y).transform(X)

    def _apply_threshold(self, X):

        # If not using fit
        if not hasattr(self, '_threshold'):
            self._threshold = self.threshold

        # Process threshold type on copy of X
        X_t = X.copy()
        if self.threshold_type == 'abs':
            X_t = np.abs(X_t)

        # If Value
        if self.threshold_method == 'value':
            if self.threshold_type == 'neg':
                return np.where(X_t <= self._threshold, 1, 0)
            return np.where(X_t >= self._threshold, 1, 0)

        elif self.threshold_method == 'density':

            # Rounded up
            top_n = round(X_t.shape[0] * X_t.shape[1] * self._threshold) - 1

            # If less than 0, set to 0
            if top_n < 0:
                top_n = 0

            # If neg, sort differently
            reverse = False if self.threshold_type == 'neg' else True
            thresh = sorted(X_t.flatten(), reverse=reverse)[top_n]

            # Neg and pos case
            if self.threshold_type == 'neg':
                return np.where(X_t <= thresh, 1, 0)
            return np.where(X_t >= thresh, 1, 0)

        raise RuntimeError(str(self.threshold_method) + ' not a valid.')

    def apply_threshold(self, X, to_nx=True):

        # Apply threshold to X
        X = self._apply_threshold(X)

        # Return either np array or nx graph
        if to_nx:
            return nx.from_numpy_array(X)
        return X

    def _incr_thresh(self):

        # Density case
        if self.threshold_method == 'density':
            self._threshold += self.ensure_incr
            return

        # Value case
        if self.threshold_type == 'neg':
            self._threshold += self.ensure_incr
        else:
            self._threshold -= self.ensure_incr

    def _lc_ratio(self, X):

        G = self.apply_threshold(X, to_nx=True)
        largest_cc = len(max(nx.connected_components(G), key=len))

        return largest_cc / G.number_of_nodes()

    def _is_conn(self, X):

        G = self.apply_threshold(X, to_nx=True)
        return nx.is_connected(G)

    def threshold_check(self, X):

        # TODO change to input check func
        # that can accept more format, vs just squeeze
        X = np.squeeze(X)

        self._threshold = self.threshold
        
        # No ensure passed
        if self.ensure is None:
            return self._threshold

        # Connected component size
        elif isinstance(self.ensure, float):

            # While the ratio is less than desired
            while self._lc_ratio(X) <= self.ensure:
                self._incr_thresh()

        # Keep incrementing until connected
        elif self.ensure == 'is_conn':
            while not self._is_conn(X):
                self._incr_thresh()

        # Make sure atleast one edge
        elif self.ensure == 'edge':
            while np.sum(self.apply_threshold(X, to_nx=False)) == 0:
                self._incr_thresh()

        else:
            raise RuntimeError('Invalid option passed for ensure.')

        
        # Return the maybe changed threshold
        return self._threshold

    def transform(self, X):
        '''Transform a passed 2D numpy correlation matrix.

        Parameters
        ----------
        X : numpy array
            A 2D numpy array representing an input correlation
            matrix.

        Returns
        ---------
        X_trans : numpy array
            Returns a flat array of length number of
            measures in parameter to_compute, representing
            the calculated network statistics.
        '''

        # TODO change to input check func
        # that can accept more format, vs just squeeze
        X = np.squeeze(X)

        # Perform threshold check based on the different settings
        self.threshold_check(X)

        # Apply threshold and cast to network
        G = self.apply_threshold(X, to_nx=True)

        # Compute metrics + keep track of feature names
        X_trans = []
        self.feat_names_ = []

        for compute in self.to_compute:
            
            # If custom function
            if compute not in self._func_dict:
                computed = compute(G)
                feat_name = compute.__name__

            # If pre-defined
            else:
                computed = self._compute(G, self._func_dict[compute])
                feat_name = compute

            # Keep track in X_trans
            X_trans += [computed]

            # If local measure
            if isinstance(computed, list):
                self.feat_names_ += [f'{feat_name}_{i}' for i in range(len(computed))]
            else:
                self.feat_names_ += [feat_name]
        
        # Stack feat name
        self.feat_names_ = np.hstack(self.feat_names_)

        # Return stacked metrics
        return np.hstack(X_trans)

    def _avg_degree(self, G):
        avg_degree = np.mean([i[1] for i in nx.degree(G)])
        return avg_degree

    def _compute(self, G, func_avg):

        # Unpack function
        func, avg = func_avg[0], func_avg[1]

        # Run
        computed = func(G)

        # If dict, then we need to either
        # average, or return as per node
        if isinstance(computed, dict):
            
            # Get as just list of values
            values = list(computed.values())
            
            # Average case
            if avg:
                return np.mean(values)
                
            return values
        
        # If single number, just return
        return computed