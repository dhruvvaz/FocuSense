import cv2
import numpy as np

# Load the model
net = cv2.dnn.readNetFromCaffe('/users/andrewvaz/Documents/deploy.prototxt', '/users/andrewvaz/Documents/res10_300x300_ssd_iter_140000.caffemodel')

# Load the image
image = cv2.imread('/users/andrewvaz/Documents/IMG-9680.jpg')

# Get the image dimensions
(h, w) = image.shape[:2]

# Preprocess the image to create a blob and perform forward pass to get the detections
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

# Loop over the detections and draw boxes around the detected faces
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Ensure the confidence is above a threshold
    if confidence > 0.5:
        # Compute the (x, y)-coordinates of the bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # Draw the bounding box of the face with the probability
        text = "{:.2f}%".format(confidence * 100)
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# Show the output image with detected faces
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


