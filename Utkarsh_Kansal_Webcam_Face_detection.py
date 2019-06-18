import cv2

# Trained XML file for detecting a face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loading WebCam
cap = cv2.VideoCapture(0)

while True:
	# Reading the video feed
    ret, img = cap.read()

    # Converting it into gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecting Face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Drawing a Rectangle around the face
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
    # Displaying the image    
    cv2.imshow('img',img)

    # Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Closing the window 
cap.release()

# De-allocating memory  
cv2.destroyAllWindows()