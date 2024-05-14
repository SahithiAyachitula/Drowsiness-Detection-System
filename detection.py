import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound('C:\\Users\\SAHITHI\\OneDrive\\Desktop\\Drowsiness\\alarm.wav')

face_cascade = cv2.CascadeClassifier('C:\\Users\\SAHITHI\\OneDrive\\Desktop\\Drowsiness\\haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('C:\\Users\\SAHITHI\\OneDrive\\Desktop\\Drowsiness\\haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('C:\\Users\\SAHITHI\\OneDrive\\Desktop\\Drowsiness\\haarcascade_righteye_2splits.xml')

model = load_model('C:\\Users\\SAHITHI\\OneDrive\\Desktop\\Drowsiness\\cnnCat2.h5')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    left_eye = leye_cascade.detectMultiScale(gray)
    right_eye = reye_cascade.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    rpred = [99]
    lpred = [99]

    for (x, y, w, h) in right_eye:
        r_eye = gray[y:y + h, x:x + w]
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255.0
        r_eye = r_eye.reshape(1, 24, 24, 1)
        rpred = np.argmax(model.predict(r_eye), axis=1)
        break

    for (x, y, w, h) in left_eye:
        l_eye = gray[y:y + h, x:x + w]
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255.0
        l_eye = l_eye.reshape(1, 24, 24, 1)
        lpred = np.argmax(model.predict(l_eye), axis=1)
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    score = max(score, 0)
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)
        try:
            sound.play()
        except:
            pass

        thicc = min(thicc + 2, 16)
        if thicc >= 16:
            thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
