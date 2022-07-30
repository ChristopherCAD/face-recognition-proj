import numpy as np
import cv2
import imutils
import pickle
import os
from PIL import Image
import PIL.Image
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer2.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture('http://192.168.1.4:8080/video')

######################################################################################

def faceRecognition():

	cap = cv2.VideoCapture('http://192.168.1.4:8080/video')

	while True:
		ret, frameWn = cap.read()


		frameWn = cv2.resize(frameWn, (720, 500))
		gray = cv2.cvtColor(frameWn, cv2.COLOR_BGR2GRAY)

		faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
		for (x, y, w, h) in faces:
			#print(x,y,w,h)
			roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
			roi_color = frameWn[y:y+h, x:x+w] 


			id_, conf = recognizer.predict(roi_gray)
			if conf>= 45: #and conf <= 85:
				#print(labels[id_])
				font = cv2.FONT_HERSHEY_SIMPLEX
				name = labels[id_]
				color = (255, 255, 255)
				stroke = 2
				cv2.putText(frameWn, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

			img_item = "faceDb\my-image.png"
			cv2.imwrite(img_item, roi_gray)

			color = (255, 0, 0)
			stroke = 2
			end_cord_x = x + w
			end_cord_y = y + h
			cv2.rectangle(frameWn, (x, y), (end_cord_x, end_cord_y), color, stroke)
		
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			roi_gray = gray[ey:ey+eh, ex:ex+ew]

		color = (0, 225, 0)
		stroke = 2
		eye_x = ex + ew
		eye_y = ey + eh
		cv2.rectangle(frameWn, (ex, ey), (eye_x, eye_y), color, stroke)

		cv2.imshow('frame', frameWn)

		if cv2.waitKey(20) & 0xFF == ord("q"):
			break

cap.release()
cv2.destroyAllWindows()

## This is all for today, let's do it again tomorrow :))
# Thank you, Sho! :DD


######################################################################################

#def create_folder():
#	source_path = filedialog.askdirectory(title = 'Select Directory')
#	path = os.path.join(source_path, 'Images')
#	os.makedirs(path)


######################################################################################

def data_trainer():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))

	image_dir = os.path.join(BASE_DIR, "faceDb")


	face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
	recognizer = cv2.face.LBPHFaceRecognizer_create()


	current_id = 0
	label_ids = {}


	y_labels = []
	x_train = []

	for root, dirs, files in os.walk(image_dir):
		for file in files:
			if file.endswith("jpg") or file.endswith("png"):
				path = os.path.join(root, file)
				label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()

				print(label, path)
				if not label in label_ids:
					label_ids[label] = current_id
					current_id += 1

				id_ = label_ids[label]
				print(label_ids)

				#y_labels.append(label)
				#x_train.append(path)
				pil_image = Image.open(path).convert("L")
				size = (550, 550)
				final_image = pil_image.resize(size, Image.ANTIALIAS)

				image_array = np.array(final_image, "uint8")

				print(image_array)

				faces = face_cascade.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5)

				for (x,y,w,h) in faces:
					roi = image_array[y:y+h, x:x+w]
					x_train.append(roi)
					y_labels.append(id_)

	#print(y_labels)
	#print(x_train)

	with open("labels.pickle", 'wb') as f:
		pickle.dump(label_ids, f)

	recognizer.train(x_train, np.array(y_labels))
	recognizer.save("trainer2.yml")



######################################################################################


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




######################################################################################


window = tk.Tk()
window.title("Face Recognition")

label = tk.Label(window, text = "FACIAL RECOGNITION", font=("Arial", 13))
label.grid(column = 0, row = 1, padx = 10, pady = 10)

face_recognition = tk.Button(window, text = "Facial Recognition", font=("Arial", 16), bg='green', fg='white', command = faceRecognition)
face_recognition.grid(column = 0, row = 2, padx = 10, pady = 10)

train_data = tk.Button(window, text = "Train Data", font=("Arial", 16), bg='green', fg='white', command = data_trainer)
train_data.grid(column = 0, row = 3, padx = 10, pady = 10)

generate_data = tk.Button(window, text = "Capture New Face-Data", font=("Arial", 16), bg='green', fg='white', command = generate_dataset)
generate_data.grid(column = 0, row = 4, padx = 10, pady = 10)





window.geometry("270x250")
window.mainloop()


