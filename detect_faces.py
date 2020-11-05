# USAGE
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()-10]

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# 获取各层信息
layer_names = net.getLayerNames()

print(layer_names)
for name in layer_names:
    id = net.getLayerId(name)
    layer = net.getLayer(id)
    print("layer id : %d, type : %s, name: %s"%(id, layer.type, layer.name))

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
print(getOutputsNames(net))
outs = net.forward(getOutputsNames(net))
# detections = net.forward("conv4_3_norm_mbox_priorbox")
print(outs)



# # loop over the detections
# for i in range(0, detections.shape[2]):
# 	# extract the confidence (i.e., probability) associated with the
# 	# prediction
# 	confidence = detections[0, 0, i, 2]

# 	# filter out weak detections by ensuring the `confidence` is
# 	# greater than the minimum confidence
# 	if confidence > args["confidence"]:
# 		# compute the (x, y)-coordinates of the bounding box for the
# 		# object
# 		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# 		(startX, startY, endX, endY) = box.astype("int")
 
# 		# draw the bounding box of the face along with the associated
# 		# probability
# 		text = "{:.2f}%".format(confidence * 100)
# 		y = startY - 10 if startY - 10 > 10 else startY + 10
# 		cv2.rectangle(image, (startX, startY), (endX, endY),
# 			(0, 0, 255), 2)
# 		cv2.putText(image, text, (startX, y),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# # show the output image
# cv2.imshow("Output", image)
# cv2.waitKey(0)