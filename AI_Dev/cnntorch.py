import torch 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim 
import torch.nn as nn 
import torch.nn.functional as F
import os
from torchsummary import summary 
import matplotlib.pyplot as plt
import numpy as np
import cv2
# from utils.plotting import *

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16, 'axes.labelweight': 'bold', 'axes.grid': False})

dataset_dir = "human_detection_dataset/"

classes = ('Human', 'None')

# Epoch 19: model accuracy 62.50% 
# https://www.tomasbeuzen.com/deep-learning-with-pytorch/chapters/chapter5_cnns-pt1.html 

class NeuralNetwork(nn.Module): 
    '''
        Convulational Neural Network 
    ''' 
    def __init__(self): 
        super().__init__()

        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=(3,3), padding = 3), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)), 
            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(2, 2), padding = 3), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)), 
            torch.nn.Flatten(), 
            torch.nn.Linear(147456, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2) 
        )

    def forward(self, data): 
        out = self.main(data)
        return F.softmax(out, dim=1) 
    
    # def optimizer(self, leaarning_rate)



def calculate_normalization(): 
    '''
        Caclulate Normalization which uses the z-score normlization
        Z-score normlization allows for easier way for cnn to conceptualize 
        and understand pixels, improving performance and gradient descent 
        
        Z-Score normlization
        output[channel] = (input[channel] - mean[channel]) / sd[channel]
    '''
    trasnform_with_normlization = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.ToTensor()
    ])

    dataset = torchvision.datasets.ImageFolder(
        root = dataset_dir, 
        transform = trasnform_with_normlization 
    )

    data_load = torch.utils.data.DataLoader(
        dataset, 
        batch_size = len(dataset), 
        shuffle = False 
    )

    mean = 0.0 
    std = 0.0 
    total_samples = 0 

    for inputs, _ in data_load: 
        batch_size = inputs.size(0)
        inputs = inputs.view(batch_size, inputs.size(1), -1)
        mean += inputs.mean(2).sum(0)
        std += inputs.std(2).sum(0)
        total_samples += batch_size 
    
    mean /= total_samples 
    std /= total_samples 

    # Mean and Standard Devivation of each pixel 
    print("Mean", mean.tolist())
    print("Standard Deviation", std.tolist())

def split_dataset(): 
    '''
    '''

    # Implement Normalization in the Transforms compose function 
    # Reszie the image 180x180 
    # Transfrom the image from [0, 255] to [0, 1]
    # Normalize by output[channel] = (input[channel] - mean[channel]) / sd[channel]

    # Mean [0.500492513179779, 0.49119654297828674, 0.4684107005596161]
    # Standard Deviation [0.19734451174736023, 0.19522404670715332, 0.19612807035446167]
    transform = transforms.Compose([
        transforms.Resize((180, 180)), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.5, 0.491, 0.468], 
            std = [0.197, 0.195, 0.196]
        )
    ])

    dataset = torchvision.datasets.ImageFolder(
        root = dataset_dir,
        transform = transform
    )

    # dataset = torch.utils.data.ConcatDataset([dataset_human, dataset_none])

    train_size = int(0.8 * len(dataset)) 
    test_size = len(dataset) - train_size

    train_set, test_set = torch.utils.data.random_split(
        dataset, 
        [train_size, test_size]
    )

    data_loader_train = torch.utils.data.DataLoader(
        train_set, 32, 
        shuffle = True, 
    )

    data_loader_validation = torch.utils.data.DataLoader(
        test_set, 32, 
        shuffle = True,
    )

    network = NeuralNetwork() 
    # summary(network, (3, 180, 180))

    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    acc = 0 
    count = 0

    # Optimize the model for 9 epochs
    n_epochs = 10


    for epoch in range(n_epochs): 
        for img, labels in data_loader_train: 
            pred = network.forward(img)
            loss = nn.CrossEntropyLoss()(pred, labels)

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
    
        with torch.no_grad(): 
            for inputs, labels in data_loader_validation: 
                y_pred = network.forward(inputs)
                acc += (torch.argmax(y_pred, 1) == labels).float().sum()
                count += len(labels)
        
        print(f"End of {epoch}, accuracy {(acc / count) * 100}")
        

    acc /= count 
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))

    torch.save(network.state_dict(), "cifar10model.pth")

            



def test_cnn(): 
    transform = transforms.Compose([
        transforms.Resize((180, 180)), 
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean = [0.5, 0.491, 0.468], 
        #     std = [0.197, 0.195, 0.196]
        # )
        # transforms.Normalize(
        #     mean = [0.5], 
        #     std = [0.197]
        # )
    ])

    dataset = torchvision.datasets.ImageFolder(
        root = dataset_dir,
        transform = transform
    )

    train_size = int(0.8 * len(dataset)) 
    test_size = len(dataset) - train_size

    train_set, test_set = torch.utils.data.random_split(
        dataset, 
        [train_size, test_size]
    )

    data_loader_train = torch.utils.data.DataLoader(
        train_set, 32, 
        shuffle = True, 
    )
    
    for img, labels in data_loader_train: 
        # image = torch.from_numpy(plt.imread("human_detection_dataset/human/1.png"))
        conv_layer = torch.nn.Conv2d(3, 3, kernel_size=(3, 3), padding = 3)
        conv_layer2 = torch.nn.MaxPool2d((2, 2))
        conv_layer3 = torch.nn.Conv2d(3, 3, kernel_size=(2, 2), padding = 3)

        img = conv_layer(img)
        img = conv_layer2(img)
        img_return = conv_layer3(img).detach()[0].numpy().T

        print(img_return)
        plt.imshow((img_return * 255).astype(np.uint8))
        plt.show()
        break



split_dataset()