import numpy as np
import cv2
import imutils
import pickle
import os
from PIL import Image
import tkinter as tk
from tkinter import messagebox
from tkinter import *
from tkinter import ttk
from tkinter import filedialog


def generate_dataset():
	face_classifier = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

	def face_crop(img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_classifier.detectMultiScale(gray, 1.3, 5)
		# scaling factor = 1.5
		# MinNeighbour = 5

		if faces is None:
			return None

		for (x,y,w,h) in faces:
			crop_faces = img[y:y+h, x:x+h]
			return crop_faces

	cap = cv2.VideoCapture('http://192.168.1.4:8080/video')
	id = 1
	img_id = 0

	while True:
		
		ret, frame = cap.read()
		if face_crop(frame) is not None:
			img_id+=1
			face = cv2.resize(face_crop(frame), (200, 200))
			face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
			#file_name_path = filedialog.askdirectory(title = 'Select Directory')
			#file_save_path = file_name_path+str(id)+"."+str(img_id)+".jpg"
			file_name_path = "faceDb/TOPHER/face."+str(id)+"."+str(img_id)+".jpg"

			cv2.imwrite(file_name_path, face)
			cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)

			cv2.imshow("Cropped Face", face)

			if cv2.waitKey(20) & 0xFF == ord("q"):
				break
	cap.release()
	cv2.destroyAllWindows()
	print("Images Collected.")

generate_dataset()

