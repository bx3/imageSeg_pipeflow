The code is developed based on jupyter notebook, python 3.5

Library sources:
skimage : conda install scikit-image  #which has randow walk, fel
GrowCut : https://github.com/nfaggian/growcut
MeanShift : https://github.com/fjean/pymeanshift
GrabCut: conda install -c menpo opencv3=3.1.0

other libraries:
scipy
PIL
scipy
matplotlib
math
numpy
time
cython

files on the same directory are (python notebooks and helper py file):
simple test: take image with true background, used for analyzing
seg_packet: use processing unit flow structure to segment image, take one image at a time
single_pass: use processing unit flow structure to segment image, process a batch

seg_helper is a helper file, place in the same directory

update 5/19/2017 Bowen Xue, contact: xbwforpc@sina.com