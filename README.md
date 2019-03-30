# Face Recognition
This project recognizes the face of the person either in video or in real-time and creates a csv file to store the name and time of the person recognized.
It uses OpenCV, dlib and face_recognition module.

## Getting started

1. Intall the requirements
```
pip install -r requirements.txt 
```

2. Create a "images" folder in the cloned directory.

3. Inside images folder, create folders by the person name to be recognized and put the images of the person to be recognized in it. Do it for every person you wannt to recognize.
    
    Note :
    You can put more than one image of the same person at different angles inside their folder.
    
4. Run encode-faces
```
python3 encode-faces.py -m <method>
``` 
</br>
      There are two method supported :
   
      a. hog : Faster but less accurate.
      b. cnn : Slower but more accurate.

- It creates a file name encodings-hog/cnn.csv in the directory that containes person name and corresponding 128-D face vector.

5. Run face-recognize.py
```
python3 face-recognize.py -m <method>
```

- It recognizes faces of known person in real-time and creates database.csv that containes name of the person and the time when they were first recognized. 
