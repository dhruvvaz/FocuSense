import cv2
import dlib

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load the image
image = cv2.imread("path_to_image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray, 0)

for face in faces:
    (startX, startY, endX, endY) = (face.left(), face.top(), face.right(), face.bottom())
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Show the output image with detected faces
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


