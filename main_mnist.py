
import dataset
from model import LeNet5, CustomMLP
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from os import chdir
chdir('C:\\Users\\eric\\Desktop')
print(os.getcwd())
# import some packages you need here
test_dir='../Desktop/test_mnist/test'
train_dir='../Desktop/train_mnist/train'

def train(model, trn_loader, device, criterion, optimizer, epoch):
    
    # write your codes here
    running_loss = 0
    acc_tmp = 0
    count = 0
    for iteration, (inputs, labels) in enumerate(trn_loader):
        # get the inputs; data is a list of [inputs, labels]    
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, outputs = torch.max(outputs, 1)
        acc_tmp += (labels == outputs).sum()
        count += len(inputs)
        print("{0} [{1}/{2}] = Loss : {3}, ACC : {4}".format(epoch, iteration, len(trn_loader), running_loss/(iteration+1), acc_tmp.item()/count))
        
    print('Finished Training')
    trn_loss = running_loss / len(trn_loader)
    acc = acc_tmp.item() / count
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    test_losses=[]
    tst_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs_, labels_ in tst_loader:
            inputs_, labels_ = inputs_.to(device), labels_.to(device)
            ouputs_ = model(inputs_)
            tst_loss=criterion(ouputs_,labels_).item()
            pred=ouputs_.data.max(1,keepdim=True)[1]
            correct += pred.eq(labels_.data.view_as(pred)).sum()
            
            
        tst_loss /= len(tst_loader.dataset)
        test_losses.append(tst_loss)
        acc_ = correct,len(tst_loader.dataset)
        print('\nTest set :Avg, loss: {:.7f}, Accuracy:{}/{} ({:.0f}%) \n'.format(tst_loss, correct ,len(tst_loader.dataset), 100. * correct / len(tst_loader.dataset)))

    return tst_loss, acc_


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    lenet=LeNet5()
    mlp=CustomMLP()
    mnist_Dataset_trn=  MNIST(train_dir)
    mnist_Dataset_tst=  MNIST(test_dir)
    trn_loader=DataLoader(mnist_Dataset_trn,batch_size=128,shuffle=True)
    tst_loader=DataLoader(mnist_Dataset_tst,batch_size=128,shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer_lenet = optim.SGD(lenet.parameters(), lr=0.001, momentum=0.9)
    optimizer_mlp= optim.SGD(mlp.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    trn_loss = []
    train_acc = []
    tst_loss = []
    test_acc = []
    test_loss, acc_ = test(lenet.to(device), tst_loader, device, criterion)
    for epoch in range(20):  
        training_loss, acc = train(lenet.to(device), trn_loader, device, criterion, optimizer_lenet, epoch)
        trn_loss.append(training_loss)
        train_acc.append(acc)
        test_loss, acc_ = test(lenet, tst_loader, device, criterion)
        tst_loss.append(test_loss)
        test_acc.append(acc_)

    trn_loss_mlp = []
    train_acc_mlp = []
    tst_loss_mlp = []
    test_acc_mlp = []
    test_loss, acc_ = test(mlp.to(device), tst_loader, device, criterion)
    for epoch in range(20):  
        training_loss, acc = train(mlp.to(device), trn_loader, device, criterion, optimizer_mlp, epoch)
        trn_loss_mlp.append(training_loss)
        train_acc_mlp.append(acc)
        test_loss, acc_ = test(mlp, tst_loader, device, criterion)
        tst_loss_mlp.append(test_loss)
        test_acc_mlp.append(acc_)
    



    fig= plt.figure()
    
    plt.plot(range(20),trn_loss,color='blue')
    plt.plot(range(20),trn_loss_mlp,color='red')
    plt.legend(['Train Loss Lent','Train Loss MLP'],loc='upper right')
    
    plt.plot(range(20),tst_loss,color='blue')
    plt.plot(range(20),tst_loss_mlp,color='red')
    plt.legend(['Test Loss Lent','Test Loss MLP'],loc='upper right')
    
    plt.plot(range(20),train_acc,color='blue')
    plt.plot(range(20),train_acc_mlp,color='red')
    plt.legend(['Train Accuracy Lent','Train Accuracy MLP'],loc='upper right')
    
    plt.plot(range(20),test_acc,color='blue')
    plt.plot(range(20),test_acc_mlp,color='red')
    plt.legend(['Test Accuracy Lent','Test Accuracy MLP'],loc='upper right')
    
if __name__ == '__main__':
    main()