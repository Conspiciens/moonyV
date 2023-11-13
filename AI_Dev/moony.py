# Project Moony
import tqdm
import random
import pathlib
import itertools
import collections
import socket
import sys
import struct
import time
import os

# For Data collection and transporting data to user 
import cv2
import einops
import numpy as np
import pickle
import netifaces 

# For training my Convolutional Neural Network 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt

class_names = ["None", "human"]

batch_size = 32
img_height = 180
img_width = 180

data_dir = "human_detection_dataset/0/*"
harddrive = ""
file_count = None 

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def build_model():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "human_detection_dataset",
        labels="inferred",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )


    val_ds = tf.keras.utils.image_dataset_from_directory(
        "human_detection_dataset",
        labels="inferred",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    class_names = sorted(os.listdir('human_detection_dataset'))
    class_names.pop(0)
    print(class_names)

    print(train_ds.class_names)

    num_classes = 5

    model = tf.keras.Sequential([
      tf.keras.layers.Rescaling(1./255),
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(5)
    ])

    model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
    )


    img = tf.keras.utils.load_img(
        "evaluation/92.png", target_size=(img_height, img_width)
    )

    plt.imshow(img)
    plt.show()


    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    print(predictions)
    score = tf.nn.softmax(predictions)
    print(np.argmax(score))


    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(train_ds.class_names[np.argmax(score)], 100 * np.max(score))
    )
    

    NUM_EPOCHS = 25
    N = NUM_EPOCHS
    print(history.history)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 3), history.history["accuracy"], label="accuracy")
    plt.plot(np.arange(0, 3), history.history["val_accuracy"], label="val_accuracy")
    plt.title("Bounding Box Regression Loss on Training Set")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    # plt.show()

    # getCamera(model)


def getCamera(model):
    # videocap = cv2.VideoCapture(0)

    while True:
        ret, frame = videocap.read()
        cv2.imshow('frame', frame)
        cv2.imwrite("frame.jpg", frame)

        img = tf.keras.utils.load_img(
            "frame.jpg", target_size=(img_height, img_width)
        )

        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


def collect_data(frame) -> None: 
    '''
       Detect people with the camera and display the rectangle within the frame  
    '''
    global file_count

    hogParams = {'winStride': (4, 4)}
    boxes, weights = hog.detectMultiScale(frame, **hogParams)

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    # We don't want rectangle in the frames 
    # for (xA, yA, xB, yB) in boxes: 
    #     cv2.rectangle(client_frame, (xA, yA), (xB, yB), (0, 255, 0), 2) 
    
    if len(boxes) > 0: 
        file_count = file_count + 1
        cv2.imwrite('personal_dataset/image' + str(file_count) + '.jpg', frame.astype('uint8'))

    return frame  


def check_for_folder() -> int: 
    '''
        Check if dataset folder is created otherwise create folder 
    '''
    FOLDER = 'personal_dataset' 

    if os.path.exists(FOLDER) == True: 
        return len(os.listdir(FOLDER))
    
    os.mkdir(FOLDER)

    return len(os.listdir(FOLDER))
        

def get_private_ip(interface='wlan0'): 
    '''
        Check if private ip address avaliable so it can wait for a client connection 
    '''
    address = netifaces.ifaddresses(interface)

    if netifaces.AF_INET in address: 
        print(address[netifaces.AF_INET][0]['addr'])
        return address[netifaces.AF_INET][0]['addr']
    else: 
        return None 

def connect_to_client(private_ip: float): 
    '''
        Begin waiting for a connection on the given priveate ip 
    '''
    PORT = 10050

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((private_ip, PORT))
    print('Listening at {}'.format(s.getsockname()))
    s.listen(10)

    return s

def server_camera():
    '''
        Server Camera that waits for connection to send the client the frame 
    '''
    global file_count 

    file_count = check_for_folder() 
    private_ip = get_private_ip() 

    # 192.168.1.25 -> Raspberry pi 4 

    # Make a socket for connection at Host and port, wait for connection
    s = None 
    if private_ip != None: 
        s = connect_to_client(private_ip)
        
    vid = cv2.VideoCapture(0)

    while True:
        if s is not None:
            conn, addr = s.accept()
            if conn: 
                while vid.isOpened():
                    img, frame = vid.read()
                    if not img:
                        break

                    a = pickle.dumps(collect_data(frame))
                    message = struct.pack("Q", len(a)) + a
                    conn.sendall(message)
                    key = cv2.waitKey(10)
                    if key == 13:
                        conn.close()
        else: 
            while vid.isOpened(): 
                img, frame = vid.read()
                if not img: 
                    break 

                collect_data(frame)
                key = cv2.waitKey(10)
                if key == 13: 
                    conn.close() 



if __name__ == '__main__':
    # server_camera()
    build_model()