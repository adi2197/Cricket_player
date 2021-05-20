import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os

# Loading the model
json_file = open("cricket_play.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("cricket_players.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories = {0: 'bhuvneshwar_kumar', 1: 'dinesh_karthik',
              2: 'hardik_pandya', 3: 'jasprit_bumrah', 
              4: 'k._l.rahul', 5: 'kedar_jadhav',
              6: 'kuldeep_yadav', 7: 'mohammed_shami', 
              8: 'ms_dhoni', 9: 'ravindra_jadeja',
              10:'rohit_sharma',11:'shikhar_dhawan',
              12:'vijay_shankar',13:'virat_kohli',14:'yuzvendra_chahal'
              }
while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    #cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    cv2.rectangle(frame, (150, 150), (400, 400), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (224, 224)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 224, 224, 3))
    prediction = {'bhuvneshwar_kumar': result[0][0], 
                  'dinesh_karthik': result[0][1], 
                  'hardik_pandya': result[0][2],
                  'jasprit_bumrah': result[0][3],
                  'k._l.rahul': result[0][4],
                  'kedhar_jadhav': result[0][5],
                  'kuldeep_yadav':result[0][6],
                  'mohammed_shami':result[0][7],
                  'ms_dhoni':result[0][8],'ravindra_jadeja':result[0][9],
                  'rohit_sharma':result[0][10],'shikhar_dhawan':result[0][11],
                  'vijay_shankar':result[0][12],'virat_kohli':result[0][13],
                  'yuzvendra_chahal':result[0][14]}
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()
