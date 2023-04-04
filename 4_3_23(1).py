import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timeit

# functions to show an image
def imshow(img):
    '''
    
    
    
    
    '''
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)),aspect='auto')
    plt.show()

    
class Net(nn.Module):
    def __init__(self):
        '''
        
        
        
        
        
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        '''
        
        
        
        
        
        
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
##Variables
Loss = []
#count = 19

##Initialization
for root_dir, cur_dir, files in os.walk(r'/home/x-dewwww/Spring_2023/BME_45000/Images'):
    count += len(files)    

    
##GPU Check
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print(device)
    

##Thread Counter
runtimes = []
threads = [1] + [t for t in range(2, 49, 2)]
for t in threads:
    torch.set_num_threads(t)
    r = timeit.timeit(setup = "import torch; x = torch.randn(1024, 1024); y = torch.randn(1024, 1024)", stmt="torch.mm(x, y)", number=5)
    runtimes.append(r)
    
plt.plot(threads,runtimes)
plt.show()
min_value = min(runtimes)
min_index = runtimes.index(min_value)
torch.set_num_threads(threads[min_index])
print(f'Using {threads[min_index]} threads')

##Initialization
data_dir = "/home/x-dewwww/Spring_2023/BME_45000/Images/"
transform = transforms.Compose([transforms.Resize(32),
transforms.CenterCrop(32),
transforms.ToTensor()])
#transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(32)])
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=count, shuffle=True)
images, labels = next(iter(dataloader))

classes = ('Not Pregnant', 'Pregnant')

# get some random training images
dataiter = iter(dataloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
net= Net()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
        Loss.append(running_loss)
print('\nFinished Training')


dataiter = iter(dataloader)
images, labels = next(dataiter)


# print images
imshow(torchvision.utils.make_grid(images))
#print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(labels.size(0))))
images, labels = images.to(device), labels.to(device)
outputs = net(images)
_, predicted = torch.max(outputs, 1)

#print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'for j in range(labels.size(0))))
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\nAccuracy of the network on the {count} test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'\nAccuracy for class: {classname:5s} is {accuracy:.1f} %')
    
plt.plot(range(len(Loss)),Loss)