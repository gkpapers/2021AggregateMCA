#!/usr/bin/env python

from argparse import ArgumentParser
import os.path as op
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from pandas.core.computation.ops import UndefinedVariableError
import pandas as pd
import numpy as np

from models import AggregateLearner


def createPipe(clf_name, nsubs):
    n_comp = 20 if nsubs > 70 else 15
    pca = PCA(n_components=n_comp)
    classifs = {
        "SVM": SVC(kernel='linear', max_iter=1e6, probability=True,
                   class_weight="balanced"),
        "RF": RandomForestClassifier(n_estimators=50, class_weight="balanced"),
        "LR": LogisticRegression(solver='liblinear', max_iter=1e6,
                                 class_weight="balanced")
    }
    pipe = Pipeline(steps=[('pca', pca), (clf_name, classifs[clf_name])])
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

    for idx, sub in enumerate(df['subject'].unique()):
        # Grab a temporary dataframe for each subject
        tdf = df.query("subject == {0} and simulation != '{1}'".format(sub, rf))
        if idx == 0:
            # First check if we are/can actually subsample this dataset
            n_sims = len(tdf['simulation'].unique())
            n_samples = np.min([n_mca, n_sims])
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
    return newdf


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument("outpath", help="Directory for storing the results.")
    parser.add_argument("dset", help="Path to H5 input data file.")
    parser.add_argument("experiment", choices=["mca", "subsample", "session"])
    parser.add_argument("target", choices=["age", "sex", "cholesterol",
                                           "rel_vo2max", "bmi"])
    # Note: "meta" aggregation includes "none"/"jackknife"
    parser.add_argument("aggregation", choices=["ref", "median", "mean",
                                                "consensus", "mega", "meta"])
    parser.add_argument("classifier", choices=["RF", "SVM", "LR"])

    parser.add_argument("--random_seed", "-r", default=41, type=int)
    parser.add_argument("--n_mca", "-n", default=20, type=int)
    parser.add_argument("--verbose", "-v", action="store_true")

    # Parse arguments, and extract details/setup experiment
    ar = parser.parse_args() if args is None else parser.parse_args(args)

    # Load dataset and create classifier
    df = pd.read_hdf(ar.dset)
    pipe = createPipe(ar.classifier, len(df['subject'].unique()))

    # If we're doing an MCA_sub experiment, subsample the dataframe.
    df = sampleSimulations(df, ar.experiment, ar.random_seed, ar.n_mca)

    # Set some parameters based on experiment type
    obs_id = "simulation" if ar.experiment == 'mca' else ar.experiment
    ref_st = "ref" if ar.experiment == 'mca' else 0
    jack = 5*ar.n_mca if ar.experiment == 'mca' else 10

    # Create aggregator object for the designed experiment
    clf = AggregateLearner(df, pipe,
                           target_id=ar.target,
                           observation_id=obs_id,
                           sample_id='subject',
                           data_id='graph',
                           refstr=ref_st,
                           cvfolds=5,
                           oos=0.2,
                           jack=jack,
                           triu=True,
                           random_seed=ar.random_seed,
                           verbose=ar.verbose)

    # Fit the model
    clf.fit(aggregation=ar.aggregation)

    # Create output file names
    experiment_pieces = [ar.experiment, ar.n_mca, ar.aggregation, ar.target,
                         ar.classifier, ar.random_seed]
    ofn = "_".join(str(e) for e in experiment_pieces)
    rep_op = op.join(ar.outpath, "report_" + ofn + ".csv")
    clf_op = op.join(ar.outpath, "clfobj_" + ofn + ".pkl")

    # Save the classification report to a CSV and the classifier(s) to a pickle
    clf.performance_report().to_csv(rep_op)
    with open(clf_op, 'wb') as fhandle:
        pickle.dump(clf, fhandle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
