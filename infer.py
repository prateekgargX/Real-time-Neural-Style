import imutils
import cv2
def forwardPass(image,net):
    image = imutils.resize(image, width=600,a = 103.939,b = 116.779,c = 123.680)
    (h, w) = image.shape[:2]

    # construct a blob from the image, set the input, and then perform a
    # forward pass of the network
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
        (a,b,c), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()

    # reshape the output tensor, add back in the mean subtraction, and
    # then swap the channel ordering
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += a
    output[1] += b
    output[2] += c
    output /= 255.0
    return output.transpose(1, 2, 0)