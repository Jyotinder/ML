from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import argparse as ap
import glob
import os
from config import *
import numpy as np

if __name__ == "__main__":
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-c', "--classifier", help="Classifier to be used", default="LIN_SVM")
    parser.add_argument('-fp',"--foldername" , help="folder name for positive feature",  required="True")
    args = vars(parser.parse_args())
    # Classifiers supported
    clf_type = args['classifier']
    fname = args['foldername']
    training_names = os.listdir(pos_feat_ph)

    fds = []
    labels = []
    # Load the positive features
    pos_feat_path=os.path.join(pos_feat_ph,fname)
    count=0
    prev= ""
    for feat_path in glob.glob(os.path.join(pos_feat_path,"*.feat")):
        fd = joblib.load(feat_path)

        if count == 0 :
            prev=fd
            count =1
        if prev.shape != fd.shape:
            print feat_path
            print fd.shape
            print fd.dtype
            continue

        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for dir in training_names:
        if fname != dir:
            neg_feat_path=os.path.join(pos_feat_ph,dir)
            for feat_path in glob.glob(os.path.join(neg_feat_path,"*.feat")):
                fd = joblib.load(feat_path)

                if count == 0 :
                    prev=fd
                    count =1
                if prev.shape != fd.shape:
                    print feat_path
                    print fd.shape
                    print fd.dtype
                    continue

                fds.append(fd)
                labels.append(0)

    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        print "Training a Linear SVM Classifier"
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        joblib.dump(clf, model_path)
        print "Classifier saved to {}".format(model_path)
