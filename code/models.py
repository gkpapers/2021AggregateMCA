#!/usr/bin/env python

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from copy import deepcopy
import pandas as pd
import numpy as np

from utils import structcon, passthrough, unstratifiedSample, refSample
from skpatch import StackingClassifier, StratifiedGroupKFold


class AggregateLearner():
    def __init__(self, dataframe, classifier, target_id, observation_id,
                 sample_id, data_id, verbose=False, oos=0.2, jack=100,
                 cvfolds=5, refstr='ref', triu=True, random_seed=42):
        # Sort by simulation ID, so "ref" is always in the same spot
        dataframe = dataframe.sort_values(observation_id)

        # Store data, classifier, and the relevant IDs
        self.df = dataframe
        self.clf_obj = classifier
        self.rs = random_seed
        self.data_id = data_id
        self.target_id = target_id
        self.observation_id = observation_id
        self.cvfolds = cvfolds
        self.verbose = verbose
        self.triu = triu
        self.n_jack = jack
        self.n_oos = oos
        self.clf = {}
        self.perf = {}
        self.oos_perf = {}

        # Sample the dataset to exclude the test set
        np.random.seed(random_seed)
        unique_samples = list(self.df[sample_id].unique())
        self.test_ids = set(np.random.choice(unique_samples,
                                             round(len(unique_samples)*oos)))
        self.train_ids = set(self.df[sample_id].unique()) - self.test_ids

        # Get samples for training (will be split into train/validate later)
        self.dat = self._grab(data_id, sample_id, self.test_ids, stack=True)
        self.sam = self._grab(sample_id, sample_id, self.test_ids)
        self.obs = self._grab(observation_id, sample_id, self.test_ids)
        self.tar = self._grab(target_id, sample_id, self.test_ids)
        self.refloc = np.where(self.obs[0] == refstr)[0]

        # Get samples for (final) testing
        self.dat_t = self._grab(data_id, sample_id, self.train_ids, stack=True)
        self.sam_t = self._grab(sample_id, sample_id, self.train_ids)
        self.obs_t = self._grab(observation_id, sample_id, self.train_ids)
        self.tar_t = self._grab(target_id, sample_id, self.train_ids)

    def fit(self, aggregation=None, *args, **kwargs):
        np.random.seed(self.rs)
        if isinstance(aggregation, str):
            aggregation = aggregation.lower()

        # Mean: take the mean of all data and train/validate/test once
        if aggregation == "mean":
            clfs, perf = self._simple_fit(func=np.mean, axis=2)
            clf, oos = self._oos_eval(clfs, func=np.mean, axis=2)

        # Median: take the median of all data and train/validate/test once
        elif aggregation == "median":
            clfs, perf = self._simple_fit(func=np.median, axis=2)
            clf, oos = self._oos_eval(clfs, func=np.median, axis=2)

        # Consensus: take the consensus of all data and train/validate/test once
        elif aggregation == "consensus":
            clfs, perf = self._simple_fit(func=structcon)
            clf, oos = self._oos_eval(clfs, func=structcon)

        # Mega: Stack all data and train/validate/test once
        elif aggregation == "mega":
            clfs, perf = self._simple_fit(func=passthrough)
            clf, oos = self._oos_eval(clfs, func=passthrough)

        # Ref: Use the reference executions
        elif aggregation == "ref":
            clfs, perf = self._simple_fit(func=refSample, index=self.refloc)
            clf, oos = self._oos_eval(clfs, func=refSample, index=self.refloc)

        # Meta: Stack "none" classifiers and train/validate once, jackknife test
        elif aggregation == "meta":
            # Ensure the jackknifed classifiers have already been fit
            if not self.clf.get("none"):
                self.fit(aggregation="none")
            clfs, perf = self._simple_fit(func=unstratifiedSample, meta=True)
            # Just wrapping in a list for the sake of the report function
            perf = [perf]
            clf, oos = self._oos_eval(clfs, func=unstratifiedSample, meta=True)

        # None/Jackknife: Sample observations and train/validate/test repeatedly
        else:
            aggregation = "none"
            clf = []
            perf = []
            oos = []
            for _ in range(self.n_jack):  # Jackknife N (=101) times
                tclfs, tperf = self._simple_fit(func=unstratifiedSample)
                tclf, toos = self._oos_eval(tclfs, func=unstratifiedSample)
                clf += [tclf]
                perf += [tperf]
                oos += [toos]

                del tclf, tperf, toos

        # Store the results as an attribute of the class object
        self.clf[aggregation] = clf
        self.perf[aggregation] = perf
        self.oos_perf[aggregation] = oos

    def _simple_fit(self, func, meta=False, *args, **kwargs):
        # Generate training data and CV object
        X, y, grp = self._prep_data(self.dat, self.tar, self.sam,
                                    func, *args, **kwargs)
        cv = StratifiedGroupKFold(n_splits=self.cvfolds)

        # Initiate data structure
        perf = {}
        perf['true'] = []
        perf['pred'] = []
        perf['acc'] = []
        perf['f1'] = []
        perf['expvar'] = []
        tmpclfs = []

        # In the case of the meta learner, pre-load classifiers from the "none"
        # setting and split them across folds.
        if meta:
            splits = self._get_folded_estimators()
            foldid = 0

        # Train and Validate model, and record in-sample performance
        for train_idx, test_idx in cv.split(X, y, grp):
            # Apply CV split to dataset
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            g_train, g_test = grp[train_idx], grp[test_idx]

            # Initialize classifier objects and fit them
            if meta:
                # Use the pre-loaded "none"-aggregation classifiers
                tmpclf = StackingClassifier(estimators=splits[foldid],
                                            passthrough=False, cv="prefit")
                foldid += 1
            else:
                tmpclf = deepcopy(self.clf_obj)
            tmpclf.fit(X_train, y_train)

            # Evaluate classifier on validation data
            pred = tmpclf.predict(X_test)
            perf['true'] += [y_test]
            perf['pred'] += [pred]
            perf['acc'] += [accuracy_score(y_test, pred)]
            perf['f1'] += [f1_score(y_test, pred)]
            # For compatible models, grab explained variance
            try:
                perf['expvar'] += [np.sum(tmpclf[0].explained_variance_ratio_)]
            except TypeError:
                perf['expvar'] += [None]

            tmpclfs += [tmpclf]
            del tmpclf

            # Print group splits and performance
            if self.verbose:
                print("Y: ", y_train, y_test)
                print("G: ", g_train, g_test)
                print("Accuracy: ", perf['acc'][-1])

        return tmpclfs, perf

    def _oos_eval(self, clfs, func, meta=False, *args, **kwargs):
        # If we're in the meta case, just call this several times regularly
        if meta:
            oos = []
            # Jackknife for proportionally fewer cases in meta eval
            for _ in range(int(np.ceil(self.n_jack*self.n_oos))):
                tmpclf, tmpoos = self._oos_eval(clfs, func, meta=False,
                                                *args, **kwargs)
                clf = tmpclf
                oos += [tmpoos]
                del tmpoos
            return clf, oos

        # Generate test / oos data
        oos = {}
        Xo, yo, grpo = self._prep_data(self.dat_t, self.tar_t, self.sam_t,
                                       func, *args, **kwargs)

        # Aggregate classifiers across folds and pre-load training
        clf = VotingClassifier(voting='soft',
                               estimators=[(i, c) for i, c in enumerate(clfs)])
        clf.estimators_ = clfs
        clf.le_ = LabelEncoder().fit(yo)
        clf.classes_ = clf.le_.classes_

        # Evaluate voting classifier on test data
        pred = clf.predict(Xo)
        oos['true'] = yo
        oos['pred'] = pred
        oos['acc'] = accuracy_score(yo, pred)
        oos['f1'] = f1_score(yo, pred)
        # Compare to mean oos-performance of component classifiers
        comp_preds = [c.predict(Xo) for c in clfs]
        oos['comp_acc'] = np.mean([accuracy_score(yo, cp) for cp in comp_preds])
        oos['comp_f1'] = np.mean([f1_score(yo, cp) for cp in comp_preds])

        # Print performance
        if self.verbose:
            print("Y: ", pred, "->", yo)
            print("G: ", grpo)
            print("Test Accuracy: ", oos['acc'])
        return clf, oos

    def _prep_data(self, data, target, group, func, *args, **kwargs):
        # Apply sampling function to the dataset and reshape the graphs to be 1D
        X = np.dstack([func(d, *args, **kwargs) for d in data])

        # If flag is set, reduce connectomes to upper triangular
        if self.triu:
            triu_ind = np.triu_indices_from(X[:,:,-1], k=1)
            Xr = np.dstack([X[..., i][triu_ind] for i in range(X.shape[-1])])
            Xr = np.reshape(Xr, (Xr.shape[1], Xr.shape[-1])).T
        else:
            Xr = np.reshape(X, (X.shape[0]**2, X.shape[2])).T

        # For IDs, there are two options: 1/brain (most) or all values (mega)
        if Xr.shape[0] == len(target):
            # In the former, take a single value (b.c. sampling)
            y = np.array([t[0] for t in target])
            grp = np.array([g[0] for g in group])
        else:
            # In the latter, flatten all
            y = np.array([_ for t in target for _ in t])
            grp = np.array([_ for s in group for _ in s])

        # Print size of resulting data structures
        if self.verbose:
            print(X.shape, Xr.shape, y.shape, grp.shape)

        return Xr, y, grp

    def _get_folded_estimators(self, aggregation="none"):
        # Turn list of classifiers across jackknives into grouped-by-fold list
        ests = [c.estimators_ for c in self.clf[aggregation]]
        splits = []
        for jdx in range(len(ests[0])):
            splits.append([])
            for est in ests:
                splits[jdx] += [est[jdx]]
            splits[jdx] = [(str(i), c) for i, c in enumerate(splits[jdx])]
        return splits

    def performance_report(self):
        # Turn separate performance structures into simple table
        dflist = []
        for k, v in self.perf.items():
            if isinstance(v, list):
                # If we have a list of lists of acc/f1 values, flatten them
                ac = [vvv for vv in v for vvv in vv['acc']]
                f1 = [vvv for vv in v for vvv in vv['f1']]

                # If we have a list of test performance values, average them
                act = np.mean([vv['acc'] for vv in self.oos_perf[k]])
                f1t = np.mean([vv['f1'] for vv in self.oos_perf[k]])

                # Treat the composite average test performance the same as above
                cact = np.mean([vv['comp_acc'] for vv in self.oos_perf[k]])
                cf1t = np.mean([vv['comp_f1'] for vv in self.oos_perf[k]])

            else:
                ac = v['acc']
                f1 = v['f1']
                act = self.oos_perf[k]["acc"]
                f1t = self.oos_perf[k]["f1"]
                cact = self.oos_perf[k]["comp_acc"]
                cf1t = self.oos_perf[k]["comp_f1"]

            # Add "row" to eventual dataframe
            dflist += [
                {
                    "aggregation": k,
                    "acc": np.mean(ac),
                    "f1": np.mean(f1),
                    "test_acc": act,
                    "test_f1": f1t,
                    "test_mean_acc": cact,
                    "test_mean_f1": cf1t,
                    "n_models": len(ac)
                }
            ]

            del ac, f1, act, f1t
        return pd.DataFrame.from_dict(dflist)

    def _grab(self, want, sweep, exclude, stack=False):
        grabbed = []
        for s in self.df[sweep].unique():
            if s in exclude:
                continue
            dat = self.df[self.df[sweep] == s][want].values
            if stack:
                dat = np.dstack(dat)
            grabbed += [dat]
        return grabbed

