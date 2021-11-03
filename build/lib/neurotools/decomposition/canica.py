"""
CanICA
"""

# Author: Alexandre Abraham, Gael Varoquaux,
# License: BSD 3 clause

import warnings as _warnings
import numpy as np

from operator import itemgetter

from scipy.stats import scoreatpercentile
from sklearn.decomposition import fastica
from joblib import Memory, delayed, Parallel
from sklearn.utils import check_random_state

from .multi_pca import MultiPCA
from nilearn._utils import fill_doc


@fill_doc
class CanICA(MultiPCA):

    def __init__(self,
                 n_components=20,
                 do_cca=True,
                 threshold='auto',
                 n_init=10,
                 random_state=None,
                 n_jobs=1, verbose=0
                 ):

        super(CanICA, self).__init__(
            n_components=n_components,
            do_cca=do_cca,
            random_state=random_state,
            n_jobs=n_jobs, verbose=verbose)

        if isinstance(threshold, float) and threshold > n_components:
            raise ValueError("Threshold must not be higher than number "
                             "of maps. "
                             "Number of maps is %s and you provided "
                             "threshold=%s" %
                             (str(n_components), str(threshold)))
        self.threshold = threshold
        self.n_init = n_init

    def _unmix_components(self, components):
        """Core function of CanICA than rotate components_ to maximize
        independence"""
        random_state = check_random_state(self.random_state)

        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        
        # Note: fastICA is very unstable, hence we use 64bit on it
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(fastica)
            (components.astype(np.float64), whiten=True, fun='cube',
             random_state=seed)
            for seed in seeds)

        ica_maps_gen_ = (result[2].T for result in results)
        ica_maps_and_sparsities = ((ica_map,
                                    np.sum(np.abs(ica_map), axis=1).max())
                                   for ica_map in ica_maps_gen_)
        ica_maps, _ = min(ica_maps_and_sparsities, key=itemgetter(-1))

        # Thresholding
        ratio = None
        if isinstance(self.threshold, float):
            ratio = self.threshold
        elif self.threshold == 'auto':
            ratio = 1.
        elif self.threshold is not None:
            raise ValueError("Threshold must be None, "
                             "'auto' or float. You provided %s." %
                             str(self.threshold))
        if ratio is not None:
            abs_ica_maps = np.abs(ica_maps)
            percentile = 100. - (100. / len(ica_maps)) * ratio
            if percentile <= 0:
                _warnings.warn("Nilearn's decomposition module "
                               "obtained a critical threshold "
                               "(= %s percentile).\n"
                               "No threshold will be applied. "
                               "Threshold should be decreased or "
                               "number of components should be adjusted." %
                               str(percentile), UserWarning, stacklevel=4)
            else:
                threshold = scoreatpercentile(abs_ica_maps, percentile)
                ica_maps[abs_ica_maps < threshold] = 0.
        # We make sure that we keep the dtype of components
        self.components_ = ica_maps.astype(self.components_.dtype)

        # flip signs in each component so that peak is +ve
        for component in self.components_:
            if component.max() < -component.min():
                component *= -1

    # Overriding MultiPCA._raw_fit overrides MultiPCA.fit behavior
    def _raw_fit(self, data):
        """Helper function that directly process unmasked data.
        Useful when called by another estimator that has already
        unmasked data.
        Parameters
        ----------
        data : ndarray or memmap
            Unmasked data to process
        """
        components = MultiPCA._raw_fit(self, data)
        self._unmix_components(components)
        return self