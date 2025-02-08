import RPi.GPIO as GPIO
import io
import picamera
import cv2
import numpy as np
from keras.models import load_model

BUZ = 4 #red led
led = 18 #green led
ALARM_ON = 23 #peizo buzzer


GPIO.setmode(GPIO.BCM)  # Use BCM GPIO numbers
GPIO.setup(BUZ, GPIO.OUT) # LED
GPIO.setup(led, GPIO.OUT) # LED
GPIO.setup(ALARM_ON, GPIO.OUT) #ALARM
GPIO.output(BUZ, GPIO.LOW) # LED
GPIO.output(led, GPIO.LOW) # LED
GPIO.output(ALARM_ON, GPIO.LOW)


#Create a memory stream so photos doesn't need to be saved in a file
stream = io.BytesIO()
#Get the picture (low resolution, so it should be quite fast)

#Load a cascade file for detecting faces
face_detection = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
left_eye_detection= cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
right_eye_detection= cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

model = load_model('models/custmodel.h5')

counter = 0
time = 0
thick = 2
right_eye_pred=[99]
left_eye_pred=[99]


while(True):
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        camera.capture(stream, format='jpeg')

    #Convert the picture into a numpy array
    buff = numpy.frombuffer(stream.getvalue(), dtype=numpy.uint8)

    #Now creates an OpenCV image
    frame = cv2.imdecode(buff, 1)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 
    #perform detection(this will return x,y coordinates , height , width of the boundary boxes object)
    faces = face_detection.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye =left_eye_detection.detectMultiScale(gray)
    right_eye =  right_eye_detection.detectMultiScale(gray)
 
 
    #iterating over faces and drawing boundary boxes for each face:
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
        
    #iterating over right eye:
    for (x,y,w,h) in right_eye:
        #pull out the right eye image from the frame:
        right_one=frame[y:y+h,x:x+w]
        counter += 1
        right_one = cv2.cvtColor(right_one,cv2.COLOR_BGR2GRAY)
        right_one = cv2.resize(right_one,(24,24))
        right_one = right_one/255
        right_one =  right_one.reshape(24,24,-1)
        right_one = np.expand_dims(right_one,axis=0)
        right_eye_pred = model.predict_classes(right_one)
        if(right_eye_pred[0] == 1):
            labels = 'Open' 
        if(right_eye_pred[0]==0):
            labels = 'Closed'
        break
 
    #iterating over left eye:
    for (x,y,w,h) in left_eye:
        #pull out the left eye image from the frame:
        left_one=frame[y:y+h,x:x+w]
        counter += 1
        left_one = cv2.cvtColor(left_one,cv2.COLOR_BGR2GRAY)  
        left_one = cv2.resize(left_one,(24,24))
        left_one = left_one/255
        left_one = left_one.reshape(24,24,-1)
        left_one = np.expand_dims(left_one,axis=0)
        left_eye_pred = model.predict_classes(left_one)
        if(left_eye_pred[0] == 1):
            labels ='Open'   
        if(left_eye_pred[0] == 0):
            labels ='Closed'
        break

#alerting part

    if(right_eye_pred[0] == 0 and left_eye_pred == 0):
        time += 1
        cv2.putText(frame,"Inactive",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(right_eye_pred[0]==1 or left_eye_pred[0]==1):
    else:
        time -= 1
        cv2.putText(frame,"Active",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(time<0):
        time=0   
        GPIO.output(ALARM_ON, GPIO.LOW)
        GPIO.output(BUZ, GPIO.LOW) # LED
        GPIO.output(led, GPIO.HIGH) # LED
      
    if(time>3):
        #person is feeling dizzy we will alert :
        GPIO.output(BUZ, GPIO.HIGH) # LED
        GPIO.output(led, GPIO.LOW) # LED
        GPIO.output(ALARM_ON, GPIO.HIGH)
               
        if(thick < 16):
            thick = thick+2
        else:
            thick=thick-2
            if(thick<2):
                thick=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thick)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
