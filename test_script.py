'''
script for test model
and predict

'''
import numpy as np
import cv2
import keras.models
from keras.models import model_from_json
from keras.optimizers import Adam
import scipy.misc


#prepare model
json_file = open("model/model.json","r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model/model.h5")
print('load_weights from disk')
print('compile model')
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

#prepare image for test
#set find face algorithm haarcascade
image = cv2.imread('output.png')

facecasc = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('gray:',gray)
faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
print('face:',faces)
if len(faces) == 0:
	print("No faces in image")
for (x, y, w, h) in faces:
	print('have face')
	roi_gray = gray[y:y + h, x:x + w]
	cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
	print('cropped_img.shape',cropped_img.shape)
	prediction = model.predict(cropped_img)
	maxindex = int(np.argmax(prediction))
	print('prediction:',emotion_dict[maxindex])
