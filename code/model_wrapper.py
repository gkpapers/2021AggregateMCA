#!/usr/bin/env python

from argparse import ArgumentParser
import os.path as op
import pickle

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from pandas.core.computation.ops import UndefinedVariableError
from scipy.stats import rankdata
import pandas as pd
import numpy as np

from models import AggregateLearner


def createPipe(embed, classif, nmca, aggregation, nsubs):
    # Dimension Reduction
    n_comp = 20 if nsubs > 70 else 15
    if embed == "pca":
        emb = ('pca', PCA(n_components=n_comp))
    else:
        emb = ('fa', FeatureAgglomeration(n_clusters=n_comp))

    # Classifiers
    neib = int(nmca*nsubs*0.1) if aggregation == "mega" else int(nsubs*0.1)
    clfs = {'svc': ('svc',
                    SVC(class_weight="balanced",
                        probability=True, max_iter=1e6)),
            'knn': ('knn',
                    KNeighborsClassifier(n_neighbors=neib)),
            'rfc': ('rfc',
                    RandomForestClassifier(class_weight="balanced")),
            'ada': ('ada',
                    AdaBoostClassifier()),
            'lrc': ('lrc',
                    LogisticRegression(class_weight="balanced",
                                       solver='liblinear', max_iter=1e6))
            }

    pipe = Pipeline(steps=[emb, clfs[classif]])
    return pipe


def sampleSimulations(df, experiment, rs, n_mca, rf="ref"):
    # Set random seed outside of loop
    np.random.seed(rs)
    # If we're evaluating multi-acquisition, remove MCA & other acq. component
    if not experiment == 'mca':
        notexp = "subsample" if experiment == "session" else "session"
        df = df.query("simulation == 'ref' and {0} == 0".format(notexp))
    else:
        try:
            df = df.query("subsample == 0 and session == 0")
        except UndefinedVariableError:
            pass  # If there aren't sessions or subsamples, use the full df

    min_nsamples = n_mca
    for idx, sub in enumerate(df['subject'].unique()):
        # Grab a temporary dataframe for each subject
        tdf = df.query("subject == {0} and simulation != '{1}'".format(sub, rf))
        # First check if we are/can actually subsample this dataset
        n_sims = len(tdf['simulation'].unique())
        n_samples = np.min([n_mca, n_sims])
        if n_samples < min_nsamples:
            min_nsamples = n_samples

        if idx == 0:
            # If we aren't subsampling at all, leave
            if n_samples == n_sims:
                newdf = df
                break

            # Otherwise, start a new dataframe with a slice of the old one
            newdf = tdf.sample(n=n_samples, axis=0)
            # Immediately add reference executions for all samples
            newdf = pd.concat([newdf,
                               df.query('simulation == "{0}"'.format(rf))])
        else:
            newdf = pd.concat([newdf, tdf.sample(n=n_samples, axis=0)])

    newdf.reset_index(inplace=True)
    return newdf, min_nsamples


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument("outpath", help="Directory for storing the results.")
    parser.add_argument("dset", help="Path to H5 input data file.")
    parser.add_argument("experiment", choices=["mca", "subsample", "session"])
    parser.add_argument("embedding", choices=["pca", "fa"])
    parser.add_argument("classifier", choices=["knn", "svc", "lrc",
                                               "rfc", "ada"])
    # Note: "meta" aggregation includes "none"/"jackknife"
    parser.add_argument("aggregation", choices=["ref", "median", "mean", "mega",
                                                "consensus", "meta", "truncate"])
    parser.add_argument("data", choices=["graph", "rankgraph",
                                         "loggraph", "zgraph"])

    parser.add_argument("--random_seed", "-r", default=41, type=int)
    parser.add_argument("--n_mca", "-n", default=20, type=int)
    parser.add_argument("--save_all", "-s", default=False, type=bool)
    parser.add_argument("--verbose", "-v", action="store_true")

    # Parse arguments, and extract details/setup experiment
    ar = parser.parse_args() if args is None else parser.parse_args(args)


    # Load dataset, and, if we're doing an MCA_sub experiment, subsample it
    df = pd.read_hdf(ar.dset)
    df, minsamples = sampleSimulations(df, ar.experiment, ar.random_seed,
                                       ar.n_mca)

    # Create classifier
    pipe = createPipe(ar.embedding, ar.classifier, ar.aggregation, minsamples,
                      len(df['subject'].unique()))

    # Set some parameters based on experiment type
    obs_id = "simulation" if ar.experiment == 'mca' else ar.experiment
    ref_st = "ref" if ar.experiment == 'mca' else 0
    jack = 5*ar.n_mca if ar.experiment == 'mca' else 10

    # Create aggregator object for the designed experiment
    clf = AggregateLearner(df, pipe,
                           target_id="age", # replace with phenotypic variable of interest
                           observation_id=obs_id,
                           sample_id='subject',
                           data_id=ar.data,
                           refstr=ref_st,
                           cvfolds=5,
                           oos=0.2,
                           jack=jack,
                           triu=True,
                           random_seed=ar.random_seed,
                           verbose=ar.verbose)

    # Fit the model
    oos = clf.fit(aggregation=ar.aggregation)

    # Create output file names
    experiment_pieces = [ar.experiment, ar.n_mca, ar.aggregation, ar.classifier,
                         ar.data, ar.embedding, ar.random_seed]
    ofn = "_".join(str(e) for e in experiment_pieces)
    rep_op = op.join(ar.outpath, "report_" + ofn + ".csv")
    clf_op = op.join(ar.outpath, "clfobj_" + ofn + ".pkl")

    # Save the classification report to a CSV and the classifier(s) to a pickle
    clf.performance_report().to_csv(rep_op)
    if ar.save_all:
        with open(clf_op, 'wb') as fhandle:
            pickle.dump(clf, fhandle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

