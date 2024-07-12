import cv2.data
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('best_model.keras')

emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.32, 3)

    for (x,y,w,h) in faces:
        sub_face_img = gray[y:y+h,x:x+w]
        print(sub_face_img)
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized/255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        print(label)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 7)
        cv2.putText(frame, emotion_labels[label], (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
    cv2.imshow('Facial expression recognition using CNN', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()