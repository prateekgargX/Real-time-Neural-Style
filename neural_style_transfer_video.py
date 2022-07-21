# import the necessary packages
from imutils.video import VideoStream
from imutils import paths
import itertools
import argparse
import infer
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
	help="path to directory containing neural style transfer models")
args = vars(ap.parse_args())

# grab the paths to all neural style transfer models in our 'models'
# directory, provided all models end with the '.t7' file extension
modelPaths = paths.list_files(args["models"], validExts=(".t7",))
modelPaths = sorted(list(modelPaths))
# generate unique IDs for each of the model paths, then combine the
# two lists together
models = list(zip(range(0, len(modelPaths)), (modelPaths)))
# use the cycle function of itertools that can loop over all model
# paths, and then when the end is reached, restart again
modelIter = itertools.cycle(models)
(modelID, modelPath) = next(modelIter)

# load the neural style transfer model from disk
print("[INFO] loading style transfer model...")
net = cv2.dnn.readNetFromTorch(modelPath)
# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
print("[INFO] {}. {}".format(modelID + 1, modelPath))

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()	
	output = infer.forwardPass(frame,net)
	# show the original frame along with the output neural style
	# transfer
	cv2.imshow("Input", frame)
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF
    	# if the `n` key is pressed (for "next"), load the next neural
	# style transfer model
	if key == ord("n"):
		# grab the next neural style transfer model model and load it
		(modelID, modelPath) = next(modelIter)
		print("[INFO] {}. {}".format(modelID + 1, modelPath))
		net = cv2.dnn.readNetFromTorch(modelPath)
	# otheriwse, if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
