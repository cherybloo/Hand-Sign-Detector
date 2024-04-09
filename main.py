import vlc
music=vlc.MediaPlayer('./indian.mp3')
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model=tf.keras.models.load_model('handGesture/mp_hand_gesture')

fhandle=open('handGesture/gesture.names','r')
labels=fhandle.read().split('\n')
fhandle.close()
print(labels)
cam=cv2.VideoCapture(0)
while True:
    success,image=cam.read()
    count=0
    width,height,_=image.shape
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    result=hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    label=''
    
    if result.multi_hand_landmarks:
        landmarks=[]
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                lmx,lmy=int(lm.x*width),int(lm.y*height)
                landmarks.append([lmx,lmy])
            mpDraw.draw_landmarks(image,hand_landmarks,mpHands.HAND_CONNECTIONS)
            prediction=model.predict([landmarks])
            id=np.argmax(prediction)
            label=labels[id]
            if(label=='call me'):
                music.play()
            elif(label=='fist'):
                music.stop()

    cv2.putText(image,label,(10,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0),2,cv2.LINE_AA)
    cv2.imshow("HELLO THERE!",image)
    if cv2.waitKey(1)==ord('d'):
        break
cam.release()
cv2.destroyAllWindows()