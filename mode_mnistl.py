
import torch.nn as nn

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """
    
    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolution
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 =nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # Max-pooling
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2) 
        # Fully connected layer
        self.fc1 = nn.Linear(16*5*5, 120)   
        self.fc2 = nn.Linear(120, 84)       
        self.fc3 = nn.Linear(84, 10)        

        

    def forward(self, img):
        # convolve, then perform ReLU non-linearity
        img = nn.functional.relu(self.conv1(img))  
        # max-pooling with 2x2 grid 
        img = self.max_pool_1(img) 
        # convolve, then perform ReLU non-linearity
        img = nn.functional.relu(self.conv2(img))
        # max-pooling with 2x2 grid
        img = self.max_pool_2(img)
        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        
        img = img.view(-1, 16*5*5)
        # FC-1, then perform ReLU non-linearity
        img = nn.functional.relu(self.fc1(img))
        # FC-2, then perform ReLU non-linearity
        img = nn.functional.relu(self.fc2(img))
        # FC-3
        output = self.fc3(img)

        

        return output

net = LeNet5()     
net.cuda()
net.parameters


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 74)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(74, 34)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(34, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, img):
        # flatten image input
        img = img.view(-1, 28*28)
        # add hidden layer, with relu activation function
        output = nn.functional.relu(self.fc1(img))
        output = nn.functional.relu(self.fc2(output))
        output = self.fc3(output)
        
        

        return output


model = CustomMLP()
model.cuda()



#check parameters 
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
pytorch1_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
