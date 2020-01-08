
# import some packages you need here
import torchvision.transforms as transforms
import os
from os import chdir
print(os.getcwd())
chdir('C:\\Users\\eric\\Desktop')
import pandas as pd
import tarfile
import glob
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import PIL
from PIL import Image


if os.path.isdir("../Desktop/train_mnist") == True:
    pass
else:
    trn_dataset = "../Desktop/train.tar"
    trn_dst_path = "../Desktop/train_mnist"
    os.makedirs(trn_dst_path)
    with tarfile.TarFile(trn_dataset,'r') as file:
        file.extractall(trn_dst_path)
    train_folder = "../Desktop/train_mnist/train"
    
    
if os.path.isdir("../Desktop/test_mnist") == True:
    pass
else:
    tst_dataset = "../Desktop/test.tar"
    tst_dst_path = "../Desktop/test_mnist"
    os.makedirs(tst_dst_path)
    with tarfile.TarFile(tst_dataset,'r') as file:
        file.extractall(tst_dst_path)
    test_folder = "../Desktop/test_mnist/test"

test_dir='../Desktop/test_mnist/test'
train_dir='../Desktop/train_mnist/train'


#왜 안될까 ################
test_set=MNIST(test_dir)
train_set=MNIST(train_dir)
#############################


class MNIST(Dataset):


    # Normalize data with mean=0.1307, std=0.3081
    
    def __init__(self, data_path):
        self.mnist_data = glob.glob(data_path)
        self.trans_form_to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.1307,), (0.3081,))
        self.data_folder = os.listdir(data_path)
        

    def __len__(self):
        return  len(self.data_folder)



    def __getitem__(self, idx):
        img_name = self.data_folder[idx]
        image = Image.open(self.mnist_data[0] + "/" + img_name)
        sample = self.trans_form_to_tensor(image)
        sample = self.normalize(sample)
        
        img = sample
        label = int(img_name[-5])

        return img, label

#if __name__ == '__main__':

    # write test codes to verify your implementations


