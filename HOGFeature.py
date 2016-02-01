import argparse as ap
from scipy.misc import imread
from skimage.feature import hog
import os
import util
from config import *
from sklearn.externals import joblib

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
parser.add_argument("-d", "--debug", help="To make debug print on",type=int, default=1)
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"]
training_names = os.listdir(train_path)
debug = args["debug"]

for dir in training_names:
    fea_path = os.path.join(pos_feat_ph, dir)
    if debug:
        print dir
    if not os.path.isdir(fea_path):
        os.makedirs(fea_path)
    set_imgpath = os.path.join(train_path, dir)
    set_img = os.listdir(set_imgpath)
    for img in set_img:
        im = util.rgb2gray(imread(os.path.join(set_imgpath, img)))
        fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
        fd_name = os.path.split(img)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(fea_path, fd_name)
        joblib.dump(fd, fd_path)
if debug:
    print "Finished Extracting HOG Feature"
