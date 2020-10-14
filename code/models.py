#!/usr/bin/env python

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_array
import numpy as np

from utils import structcon, passthrough


class AggregatedLearner():
    def __init__(self, dataframe, classifier, target_id, observation_id,
                 sample_id, data_id, hold=0.1, cvfolds=10, random_seed=42):
        # Store data, classifier, and the relevant IDs
        self.df = dataframe
        self.clf = classifier
        self.rs = random_seed
        self.data_id = data_id
        self.target_id = target_id
        self.observation_id = observation_id
        self.cvfolds = cvfolds
        self.performance = {}

        # Sample the dataset to exclude the test set
        np.random.seed(random_seed)
        unique_samples = list(self.df[sample_id].unique())
        self.test_ids = set(np.random.choice(unique_samples,
                                             round(len(unique_samples)*hold)))
        self.train_ids = set(self.df[sample_id].unique()) - self.test_ids

        # Get samples for training (will be split into train/validate later)
        self.dat = self._grab(data_id, sample_id, self.test_ids, stack=True)
        self.sam = self._grab(sample_id, sample_id, self.test_ids)
        self.obs = self._grab(observation_id, sample_id, self.test_ids)
        self.tar = self._grab(target_id, sample_id, self.test_ids)

        # Get samples for (final) testing
        self.dat_t = self._grab(data_id, sample_id, self.train_ids, stack=True)
        self.obs_t = self._grab(observation_id, sample_id, self.train_ids)
        self.tar_t = self._grab(target_id, sample_id, self.train_ids)

    def fit(self, aggregation=None, *args, **kwargs):
        np.random.seed(self.rs)
        if isinstance(aggregation, str):
            aggregation = aggregation.lower()

        if aggregation == "mean":
            self.performance[aggregation] = self._simple_fit(func=np.mean,
                                                             axis=2)
        elif aggregation == "median":
            self.performance[aggregation] = self._simple_fit(func=np.median,
                                                             axis=2)
        elif aggregation == "consensus":
            self.performance[aggregation] = self._simple_fit(func=structcon)
        elif aggregation == "mega":
            self.performance[aggregation] = self._simple_fit(func=passthrough)
        elif aggregation == "meta":
            self.performance[aggregation] = self._meta_fit()
        else:  # None
            #TODO: decide how many loops to jackknife
            perf = []
            for _ in range(101):  # Jackknife 101 times
                perf += [self._simple_fit(func=unstratifiedSample)]
            self.performance["none"] = perf

    def _simple_fit(self, func=np.mean, *args, **kwargs):
        X = np.dstack([func(d, *args, **kwargs) for d in self.dat])
        Xr = np.reshape(X, (X.shape[0]**2, X.shape[2])).T
        # For IDs, there are two options: all values (mega) or 1/brain (others)
        if X.shape[2] == len(self.train_ids):
            y = np.array([t[0] for t in self.tar])
            grp = np.array([s[0] for s in self.sam])

        else:
            y = np.array([_ for t in self.tar for _ in t])
            grp = np.array([_ for s in self.sam for _ in s])

        cv = StratifiedGroupKFold(n_splits=self.cvfolds)

        perf = {}
        perf['score'] = []
        perf['targ'] = []
        perf['pred'] = []
        for train_idx, test_idx in cv.split(Xr, y, grp):
            X_train, X_test = Xr[train_idx], Xr[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            self.clf.fit(X_train, y_train)
            perf['targ'] += [y_test]
            pred = self.clf.predict(X_test)
            perf['pred'] += [pred]
            perf['score'] += self.clf.score(X_test, y_test)
        return perf

    def _grab(self, want, sweep, exclude, stack=False):
        grabbed = []
        for s in self.df[sweep].unique():
            if s in exclude:
                continue
            dat = self.df.query("{0} == '{1}'".format(sweep, s))[want].values
            if stack:
                dat = np.dstack(dat)
            grabbed += [dat]
        return grabbed


# Class patch for sklearn to enable Stratified Group K-Fold CV
# Lifted from: https://github.com/scikit-learn/scikit-learn/pull/15239/files
class StratifiedGroupKFold(StratifiedKFold):
    """Stratified K-Folds iterator variant with non-overlapping groups.
    This cross-validation object is a variation of StratifiedKFold that returns
    folds stratified by group class. The folds are made by preserving the
    percentage of groups for each class.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    The difference between GroupKFold and StratifiedGroupKFold is that
    the former attempts to create balanced folds such that the number of
    distinct groups is approximately the same in each fold, whereas
    StratifiedGroupKFold attempts to create folds which preserve the
    percentage of groups for each class.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedGroupKFold
    >>> X = np.ones((17, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    >>> cv = StratifiedGroupKFold(n_splits=3)
    >>> for train_idxs, test_idxs in cv.split(X, y, groups):
    ...     print("TRAIN:", groups[train_idxs])
    ...     print("      ", y[train_idxs])
    ...     print(" TEST:", groups[test_idxs])
    ...     print("      ", y[test_idxs])
    TRAIN: [3 3 3 4 6 6 7 8 8]
           [1 1 1 1 0 0 0 0 0]
     TEST: [1 1 2 2 5 5 5 5]
           [0 0 1 1 0 0 0 0]
    TRAIN: [1 1 2 2 4 5 5 5 5 8 8]
           [0 0 1 1 1 0 0 0 0 0 0]
     TEST: [3 3 3 6 6 7]
           [1 1 1 0 0 0]
    TRAIN: [1 1 2 2 3 3 3 5 5 5 5 6 6 7]
           [0 0 1 1 1 1 1 0 0 0 0 0 0 0]
     TEST: [4 8 8]
           [1 0 0]
    >>> cv = GroupKFold(n_splits=3)
    >>> for train_idxs, test_idxs in cv.split(X, y, groups):
    ...     print("TRAIN:", groups[train_idxs])
    ...     print("      ", y[train_idxs])
    ...     print(" TEST:", groups[test_idxs])
    ...     print("      ", y[test_idxs])
    TRAIN: [2 2 3 3 3 4 6 6 7 8 8]
           [1 1 1 1 1 1 0 0 0 0 0]
     TEST: [1 1 5 5 5 5]
           [0 0 0 0 0 0]
    TRAIN: [1 1 5 5 5 5 6 6 7 8 8]
           [0 0 0 0 0 0 0 0 0 0 0]
     TEST: [2 2 3 3 3 4]
           [1 1 1 1 1 1]
    TRAIN: [1 1 2 2 3 3 3 4 5 5 5 5]
           [0 0 1 1 1 1 1 1 0 0 0 0]
     TEST: [6 6 7 8 8]
           [0 0 0 0 0]
    Notes
    -----
    The implementation is designed to:
    * Generate test sets such that all contain the same distribution of
      group classes, or as close as possible.
    * Be invariant to class label: relabelling ``y = ["Happy", "Sad"]`` to
      ``y = [1, 0]`` should not change the indices generated.
    * Preserve order dependencies in the dataset ordering, when
      ``shuffle=False``: all samples from class k in some test set were
      contiguous in y, or separated in y by samples from classes other than k.
    * Generate test sets where the smallest and largest differ by at most one
      group.
    See also
    --------
    StratifiedKFold: Takes class information into account to build folds which
        retain class distributions (for binary or multiclass classification
        tasks).
    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def _iter_test_masks(self, X, y, groups):
        y = check_array(y, ensure_2d=False, dtype=None)
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)
        (unique_groups, unique_groups_y), group_indices = np.unique(
            np.stack((groups, y)), axis=1, return_inverse=True)
        n_groups = len(unique_groups)
        if self.n_splits > n_groups:
            raise ValueError("Cannot have number of splits n_splits=%d greater"
                             " than the number of groups: %d."
                             % (self.n_splits, n_groups))
        if unique_groups.shape[0] != np.unique(groups).shape[0]:
            raise ValueError("Members of each group must all be of the same "
                             "class.")
        for group_test in super()._iter_test_masks(X=unique_groups,
                                                   y=unique_groups_y):
            # this is the mask of unique_groups in the partition invert it into
            # a data mask
            yield np.in1d(group_indices, np.where(group_test))
