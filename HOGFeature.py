import argparse as ap
from scipy.misc import imread
from skimage.feature import hog
import os
import util
from config import *
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from skimage import data, color, exposure
import cv2

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set")
parser.add_argument("-d", "--debug", help="To make debug print on", type=int, default=1)
parser.add_argument('-i', "--image", help="Path to the test image",default="")
args = vars(parser.parse_args())


debug = args["debug"]
image = args["image"]
if not image:
    # Get the training classes names and store them in a list
    train_path = args["trainingSet"]
    training_names = os.listdir(train_path)
    for dir in training_names:
        fea_path = os.path.join(pos_feat_ph, dir)
        if debug:
            print dir
        if not os.path.isdir(fea_path):
            os.makedirs(fea_path)
        set_imgpath = os.path.join(train_path, dir)
        set_img = os.listdir(set_imgpath)
        for img in set_img:
            #im = util.rgb2gray(imread(os.path.join(set_imgpath, img)))
            im = cv2.imread(os.path.join(set_imgpath, img))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd_name = os.path.split(img)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(fea_path, fd_name)
            joblib.dump(fd, fd_path)
    if debug:
        print "Finished Extracting HOG Feature"

else:
    im = cv2.imread(image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #fd, hog_image = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
    fd, hog_image = hog(im, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(3, 3), visualise=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(im, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()
