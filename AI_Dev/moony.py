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
import threading

# For Data collection and transporting data to user 
import cv2
import einops
import numpy as np
import pickle
import netifaces 
from sys import platform

# For training my Convolutional Neural Network 
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim 
import torch.nn as nn 
import matplotlib.pyplot as plt

class_names = ["None", "human"]

batch_size = 32
img_height = 180
img_width = 180

data_dir = "human_detection_dataset/0/*"
harddrive = ""
file_count = None 

sock = None
addr = None 
conn = None


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


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
        

def get_private_ip(interface='wlan0') -> str: 
    '''
        Check if private ip address avaliable so it can wait for a client connection 
    '''
    address = None 
    if platform == "linux" or platform == "linux2": 
        address = netifaces.ifaddresses(interface)
    elif platform == "darwin": 
        interface = 'en0'
        address = netifaces.ifaddresses(interface)

    if netifaces.AF_INET in address: 
        print(address[netifaces.AF_INET][0]['addr'])
        return str(address[netifaces.AF_INET][0]['addr'])
    else: 
        return ""

def connect_to_client(private_ip: str) -> None: 
    '''
        Begin waiting for a connection on the given private ip 
         
    '''
    global sock
    global conn
    global addr
    PORT = 10050

    try: 
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(( private_ip, PORT ))
        print('Listening at {}'.format(sock.getsockname()))
        sock.listen(2)
        conn, addr = sock.accept() 

        # Removed sock as global for multi-threading i/o task
        # return sock

    except socket.error as e: 
        raise

def server_camera() -> None:
    '''
        Server Camera that waits for connection to send the client the frame 
    '''
    global file_count 
    global sock

    file_count = check_for_folder() 
    private_ip = get_private_ip() 

    # 192.168.1.25 -> Raspberry pi 4 

    # Make a socket for connection at Host and port, wait for connection
    proceed = True
    conn = None 
    addr = None 
    clientthread = None
    try: 
        if private_ip:
            clientthread = threading.Thread(None, connect_to_client, args=(private_ip,))
            clientthread.start()
            print("Client thread is starting")

        vid = cv2.VideoCapture(0)

        if (clientthread is not None 
            and clientthread.is_alive() == False):
            clientthread.join()
        
        handle_client(vid, clientthread)
    except Exception as e: 
        print(f"Error: {e}") 
    finally: 
        if sock is not None: 
            sock.close()
    


def handle_client(vid, clientthread): 
    global sock 
    global conn 
    global addr

    while vid.isOpened(): 
        img, frame = vid.read() 
        if not img: 
            break 
        
        collect_data(frame)

        if conn: 
            a = pickle.dumps(collect_data(frame))
            message = struct.pack("Q", len(a)) + a
            conn.sendall(message)
            key = cv2.waitKey(13)
            if key == 13: 
                conn.close() 
    
        if (conn == None 
        and clientthread is not None  
        and clientthread.is_alive() == False): 
            private_ip = get_private_ip()
            if not private_ip: 
                continue
            sock = connect_to_client(private_ip)
            clientthread.join()
            if sock is None: 
                continue
            conn, addr = sock.accept() 


if __name__ == '__main__':
    server_camera()
    # build_model()
