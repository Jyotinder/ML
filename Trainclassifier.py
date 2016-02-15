from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import argparse as ap
import glob
import os
from config import *
import numpy as np
from sklearn import linear_model

def sav_SVM(fname,clf_type,training_names):
    fds = []
    labels = []
    # Load the positive features
    max=0
    pos_feat_path=os.path.join(pos_feat_ph,fname)
    for feat_path in glob.glob(os.path.join(pos_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        if fd.shape[0]>max:
            max=fd.shape[0]

        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for dir in training_names:
        if fname != dir:
            neg_feat_path=os.path.join(pos_feat_ph,dir)
            for feat_path in glob.glob(os.path.join(neg_feat_path,"*.feat")):
                fd = joblib.load(feat_path)
                if fd.shape[0]>max:
                    max=fd.shape[0]
                fds.append(fd)
                labels.append(0)
    for index, item in enumerate(fds):
        if item.shape[0]<max:
            c=np.zeros(max)
            c[:item.shape[0]]=item
            fds[index]=c
    count1=0
    if count1 ==0:
        f = open('./data/config/max.txt', 'w')
        f.write(str(max))
        f.close()
        count1+=1

    classes = np.unique(training_names)

    if clf_type is "LIN_SVM":
        #clf = LinearSVC()
        clf =linear_model.SGDClassifier()
        print "Training a Linear SVM Classifier"
        clf.partial_fit(fds, labels,classes)
        # If feature directories don't exist, create them
        path=model_path
        if not os.path.isdir(path):
            os.makedirs(path)
        path=os.path.join(path,fname)
        if not os.path.isdir(path):
            os.makedirs(path)
        path=os.path.join(path,"svm.model")
        joblib.dump(clf, path)
        print "Classifier saved to {}".format(model_path)


if __name__ == "__main__":
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-c', "--classifier", help="Classifier to be used", default="LIN_SVM")
    parser.add_argument('-fp',"--foldername" , help="folder name for positive feature")
    parser.add_argument('-all',"--allfoldername" , help="Creating SVM for each of the dataset", type=int, default=0)
    args = vars(parser.parse_args())
    # Classifiers supported
    clf_type = args['classifier']
    allF = args['allfoldername']
    training_names = os.listdir(pos_feat_ph)
    if not allF:
        fname = args['foldername']
        sav_SVM(fname,clf_type,training_names)
    else:
        for dir in training_names:
            sav_SVM(dir,clf_type,training_names)
