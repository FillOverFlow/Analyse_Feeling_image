#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
from keras.models import model_from_json
from keras.optimizers import Adam
#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re
import base64
#system level operations (like loading files)
import sys
#for reading operating system data
import os
import cv2
from model.load_data import *
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))


#initalize our flask app


app = Flask(__name__)
#global vars for easy reusability
global model,graph
#initialize these variables
model, graph = init()



#decoding an image from base64 into raw representation
def convertImage(imgData1):

	imgstr = re.search(r'base64,(.*)',str(imgData1)).group(1)
	print('imagestr:',imgstr)
	print (type(imgstr))
	with open('output.png','wb') as output:
		output.write(base64.b64decode(imgstr))


@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	#whenever the predict method is called, we're going
	#to input the user drawn character as an image into the model
	#perform inference, and return the classification
	#get the raw data format of the image

	print('in prediction...')
	imgData = request.get_data()
	#encode it into a suitable format
	convertImage(imgData)

	print ("debug")
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
	with graph.as_default():
		#perform the prediction
		prediction = model.predict(cropped_img)

		maxindex = int(np.argmax(prediction))
		#convert the response to a string
		response = emotion_dict[maxindex]
		print('prodict_image:',response)
		return response


if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 7793))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)
