#!/usr/bin/env python

from argparse import ArgumentParser
import os.path as op
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import pandas as pd
import numpy as np

from models import AggregateLearner


def createPipe(clf_name, experiment):
    n_comp = 20 if experiment.startswith('mca') else 15
    pca = PCA(n_components=n_comp)
    classifs = {
        "SVM": SVC(kernel='linear', probability=True),
        "RF": RandomForestClassifier(),
        "LR": LogisticRegression(solver='liblinear')
    }
    pipe = Pipeline(steps=[('pca', pca), (clf_name, classifs[clf_name])])
    return pipe


def sampleSimulations(df, rs, n_mca):
    # Set random seed outside of loop
    np.random.seed(rs)
    for idx, sub in enumerate(df['subject'].unique()):
        # Grab a temporary dataframe for each subject
        tdf = df.query("subject == {0}".format(sub))
        # Then, either create or add to the a dataframe with a subset
        if idx == 0:
            newdf = tdf.sample(n=n_mca, axis=0)
        else:
            newdf = pd.concat([newdf, tdf.sample(n=n_mca, axis=0)])
    return newdf


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument("outpath", help="Directory for storing the results.")
    parser.add_argument("dset", help="Path to H5 input data file.")
    parser.add_argument("experiment", choices=["mca_total", "mca_sub",
                                               "sample", "session"])
    parser.add_argument("target", choices=["age", "sex", "cholesterol",
                                           "rel_vo2max", "bmi"])
    # Note: "meta" aggregation includes "none"/"jackknife"
    parser.add_argument("aggregation", choices=["ref", "median", "mean",
                                                "consensus", "mega", "meta"])
    parser.add_argument("classifier", choices=["RF", "SVM", "LR"])

    parser.add_argument("--random_seed", "-r", default=41, type=int)
    parser.add_argument("--n_mca", "-n", choices=[15, 10, 5, 2], default=15)
    parser.add_argument("--verbose", "-v")

    # Parse arguments, and extract details/setup experiment
    ar = parser.parse_args() if args is None else parser.parse_args(args)

    # Load dataset and create classifier
    df = pd.read_hdf(ar.dset)
    pipe = createPipe(ar.classifier, ar.experiment)

    # If we're doing an MCA_sub experiment, subsample the dataframe.
    df = sampleSimulations(df, ar.random_seed, ar.n_mca)

    # Set some parameters based on experiment type
    obs_id = "simulation" if ar.experiment.startswith('mca') else ar.experiment
    ref_st = "ref" if ar.experiments.startswith('mca') else 0
    jack = 100 if ar.experiments.startswith('mca') else 10

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
    experiment_pieces = [ar.experiment, ar.aggregation, ar.target,
                         ar.classifier, ar.random_seed]
    ofn = "_".join(experiment_pieces)
    rep_op = op.join(ar.outpath, "report_" + ofn + ".csv")
    clf_op = op.join(ar.outpath, "clfobj_" + ofn + ".pkl")

    # Save the classification report to a CSV and the classifier(s) to a pickle
    clf.performance_report().to_csv(csv_op)
    with open(clf_op, 'rb') as fhandle:
        pickle.dump(clf, fhandle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

