import numpy as np
from scipy.misc import imread
from os import listdir, getcwd
from os.path import isfile, join, basename
from sklearn import svm
from skimage.feature import hog
from sklearn.svm import LinearSVC

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def read_dir(path):
    lable=""
    X=[]
    Y=[]
    count=0
    for f in listdir(path):
        if not isfile(f):
            lable= basename(f)
            dir=join(path,lable)
            onlyfiles = [j for j in listdir(dir) if isfile(join(dir,j))]
            for i in onlyfiles:
                print(join(dir,i))
                img = imread(join(dir,i))
                gray = rgb2gray(img)
                fd, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                                    cells_per_block=(1, 1), visualise=True)
                print(fd)
                X.append(fd)
                Y.append(count)
            count=count +1
    X=np.array(X)
    Y=np.array(Y)
    return (X,Y)

def main():
    path = join(getcwd(),"UCMerced_LandUse")
    path = join(path,"Images")
    data,label = read_dir(path)
    #clf = svm.SVC(kernel='linear', C = 1.0)
    #X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    #y = np.array([1, 1, 2, 2])
    clf = LinearSVC(C=1)

    #clf.fit(X, y)
    clf.fit(data,label)
    print(clf.predict(data[-10]))

if __name__=='__main__':
    main()