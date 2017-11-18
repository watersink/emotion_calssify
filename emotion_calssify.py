import cv2
import time
import os
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

"""network
inputs = Input(shape=( 48 ,48,1))
Convolution_1=Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(inputs) 
Convolution_2=Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(Convolution_1) 
Convolution_3=Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(Convolution_2)
MaxPooling_1=MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(Convolution_3)

Convolution_4=Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(MaxPooling_1) 
Convolution_5=Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(Convolution_4) 
Convolution_6=Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(Convolution_5)
MaxPooling_2=MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(Convolution_6)

Convolution_7=Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(MaxPooling_2) 
Convolution_8=Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(Convolution_7) 
Convolution_9=Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(Convolution_8)
MaxPooling_3=MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(Convolution_9)

Flatten_1 = Flatten()(MaxPooling_3)
Dense_1=Dense(256, activation='relu')(Flatten_1)
Dropout_1=Dropout(0.3)(Dense_1)
Dense_2=Dense(256, activation='relu')(Dropout_1)
Dropout_2=Dropout(0.3)(Dense_2)
Dense_3=Dense(6, activation='softmax')(Dropout_2)

"""


class Emotion_analysis(object):
    def __init__(self):
        self.emotion_labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.cascPath = "haarcascade_frontalface_default.xml"
        self.modelpath="my.hdf5"
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.model=model=load_model(self.modelpath)

        self.y_angry = []
        self.y_fear = []
        self.y_happy = []
        self.y_sad = []
        self.y_surprise = []
        self.y_neutral=[]
        self.xs=[]

        self.fig = plt.figure()
        self.ax=self.fig.add_subplot(1, 1, 1)

    def predict_emotion(self,face_image):
        face_image_gray=cv2.cvtColor(face_image,cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(face_image_gray, (48, 48), interpolation=cv2.INTER_AREA)
        image = resized_img.reshape(1, 1, 48, 48)
        list_of_list = self.model.predict(image, batch_size=1, verbose=1)
        return list_of_list[0]

    def animate_append(self,list_of_list):
        self.y_angry.append(list_of_list[0])
        self.y_fear.append(list_of_list[1])
        self.y_happy.append(list_of_list[2])
        self.y_sad.append(list_of_list[3])
        self.y_surprise.append(list_of_list[4])
        self.y_neutral.append(list_of_list[5])
        self.xs.append(time.time())

    def animate(self):
        self.ax.clear()
        self.ax.plot(self.xs, self.y_angry, label="angry")
        self.ax.plot(self.xs, self.y_fear, label="fear")
        self.ax.plot(self.xs, self.y_happy, label="happy")
        self.ax.plot(self.xs, self.y_sad, label="sad")
        self.ax.plot(self.xs, self.y_surprise, label="surprise")
        self.ax.plot(self.xs, self.y_neutral, label="neutral")
        self.ax.legend(loc='upper left')
        plt.show()

    def image_analysis(self, image_name):
        face_image = cv2.imread(image_name)
        list_of_list = self.predict_emotion(face_image)
        outresult = self.emotion_labels[int(np.argmax(list_of_list))]
        print(image_name[:-3], ":", outresult)

    def image_detect_analysis(self,image_name):
        img=cv2.imread(image_name)
        faces = self.faceCascade.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=0
        )
        for face in faces:
            face_image = img[face[1]:face[1] + face[3], face[2]:faces[2]+face[4]]
            list_of_list = self.predict_emotion(face_image)
            outresult = self.emotion_labels[int(np.argmax(list_of_list))]
            print(image_name[:-3],":",outresult)

    def video_analysis(self,videoname,show_animate):
        video_capture = cv2.VideoCapture(videoname)
        while True:
            ret, frame = video_capture.read()
            if frame is None:
                break
            faces = self.faceCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=0
            )

            for (x, y, w, h) in faces:
                face_image = frame[y:y + h, x:x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                list_of_list = self.predict_emotion(face_image)
                outresult = self.emotion_labels[int(np.argmax(list_of_list))]
                print(outresult)
                if show_animate == True:
                    self.animate_append(list_of_list)
                    print(list_of_list)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        video_capture.release()
        cv2.destroyAllWindows()
        if show_animate == True:
            self.animate()



if __name__=="__main__":
    """video test"""
    #ea = Emotion_analysis()
    #ea.video_analysis("ft1.mp4", True)

    """image test"""
    ea = Emotion_analysis()
    base_dir="./faces/"
    for img_dir in os.listdir("./faces/"):
        ea.image_analysis(os.path.join(base_dir,img_dir))



