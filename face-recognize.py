import argparse
import pandas as pd
import numpy as np
import cv2
import ast
import csv
import face_recognition
from datetime import datetime
from pytz import timezone
import os

# command line argumemts
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', type = str, default = 'hog',
                    choices = ['hog', 'cnn'], help = 'mehtod to use to detect faces')
parser.add_argument('-v','--video', help = 'input the video to recognize faces')

args = parser.parse_args()

# capture video
cap = cv2.VideoCapture(args.video if args.video != None else -1)

known_encodings = []

if args.method == 'hog' :
    with open('encodings-hog.csv', 'r') as db :
        hog_csv = pd.read_csv(db)
        encodes = hog_csv['Encodings']
        known_names = hog_csv['Name'].tolist()
        for value in encodes :
            known_encodings.append(ast.literal_eval(value))

if args.method == 'cnn' :
    with open('encodings-cnn.csv', 'r') as db :
        cnn_csv = pd.read_csv(db)
        encodes = cnn_csv['Encodings']
        known_names = hog_csv['Name'].tolist()
        for value in encodes :
            known_encodings.append(ast.literal_eval(value))

# open database to save person time
file = open(os.getcwd() + '/database.csv', 'a') 
writer = csv.writer(file)
writer.writerow(['Name', 'Time'])

all_names = {}

while True :
    ret, img = cap.read()

    if ret == False :
        print("[-] Could not read the frame")
        break

    img = cv2.flip(img,1)
    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(rgb_img,1,model = args.method)
    encodings = face_recognition.face_encodings(rgb_img,faces,1)

    recognized_names = []
    
    for encoding in encodings :
        matches = face_recognition.compare_faces(known_encodings,encoding,tolerance=0.5)
        name = 'Unknown'
        
        if True in matches :
            ids = np.argwhere(matches)
            
            for i in ids :
                if known_names[i[0]] not in recognized_names :
                    recognized_names.append(known_names[i[0]])
                if known_names[i[0]] not in all_names.keys() :
                    tz = timezone('Asia/Kolkata')
                    now = datetime.now(tz)
                    time = now.strftime('%H-%M-%S')

                    all_names[known_names[i[0]]]=time
                    writer.writerow([known_names[i[0]],time])
                    
        for ((t,r,b,l),n) in zip(faces,recognized_names) :
            cv2.rectangle(img, (l,t), (r,b), (0,0,255), 1)
            cv2.rectangle(img, (l, b - 25), (r, b), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, n, (l + 6, b - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
       
    cv2.imshow('Face-recognition',img)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        cv2.destroyAllWindows()
        file.close()
        break

cap.release()

