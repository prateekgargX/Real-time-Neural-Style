# imports
import argparse
import infer
import cv2
import os

# argument parser for the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="neural style transfer model")
ap.add_argument("-i", "--image", required=True,
	help="input image to apply neural style transfer to")
ap.add_argument("-p", "--path", required=False,
	help="path/name of the output image")
args = vars(ap.parse_args())

net = cv2.dnn.readNetFromTorch(args["model"])
image = cv2.imread(args["image"])
output = infer.forwardPass(image,net)

# save and show the images
basepath = 'images/outputs/'
if not os.path.exists(basepath): os.mkdir(basepath)
savepath = basepath+args["model"].split('/')[-1].split('.')[0]+'-'+args["image"].split('/')[-1]
if not args["path"]==None : savepath = args["path"]
cv2.imwrite(savepath, 255*output)
cv2.waitKey(0)