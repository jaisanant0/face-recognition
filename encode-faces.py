# network architecture for face recognition is based on ResNet-34.
# the network is trained by Davis King on LFW having 99.38% accuracy.
import face_recognition
import os
import glob
import argparse
import cv2
import csv
import pandas as pd
import numpy as np

# command line argument
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', type=str, choices = ['hog', 'cnn'], default = 'hog',
                    help = 'method to use to detect faces in image')

args = parser.parse_args()

print("[+] Using method : " + args.method)

# path of person images 
images_path = os.getcwd() + '/images/'
persons = os.listdir(images_path)

# list to save name and encodings
encodings = []
names = []

for person in persons :
    for person_img in glob.glob(images_path + person + '/' + '*.jpg') :
        img_name = os.path.basename(person_img).split('.')[0]

        print("[+] Creating face encoding for " + person + ' and  image : ' + img_name)
        img_read = cv2.imread(person_img)
        img_rgb = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)

        # find the dimensions of box cintaining faces
        faces = face_recognition.face_locations(img_rgb,2,model = args.method)

        # check if face found
        if len(faces) != 0 :
            # create face encodings
            encoding = face_recognition.face_encodings(img_rgb,faces,2)
            encodings.append(np.array(*encoding).tolist())
            names.append(person)
            
        # if faces are not found
        else :
            print("[-] Could not find face for " + person + 'and image : ' + img_name)
            
# if method is HOG
if args.method == 'hog' :
    with open(os.getcwd() + '/encodings-hog.csv',mode = 'a') as encode_db :
        writer = csv.writer(encode_db)
        writer.writerow(['Name', 'Encodings'])

        for n,e in zip(names,encodings) :
            writer.writerow([n,e])
       
# if method is cnn
if args.method == 'cnn' :
    with open(os.getcwd() + '/encodings-cnn.csv',mode = 'a') as encode_db :
        writer = csv.writer(encode_db)
        writer.writerow(['Name', 'Encodings'])

        for n,e in zip(names,encodings) :
            writer.writerow([n,e])
