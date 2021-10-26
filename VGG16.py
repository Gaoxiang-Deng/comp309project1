# %%
import matplotlib
import torch
import torchvision

from torch import classes
from torchvision import transforms

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.RandomHorizontalFlip(),
     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
     transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transform_test = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.RandomHorizontalFlip(),
     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
     transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

classes = ('cherry', 'strawberry', 'tomato')

data_path = "C:/workspace/comp309project/traindata"
data_all = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
test_data_path = "C:/workspace/comp309project/testdata"
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform_test)

# %%
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler


def dataset_sampler(dataset, val_percentage=0.1):
    """
    split dataset into train set and val set
    :param dataset:
    :param val_percentage: validation percentage
    :return: split sampler
    """
    sample_num = len(dataset)
    file_idx = list(range(sample_num))
    train_idx, val_idx = train_test_split(file_idx, test_size=val_percentage, random_state=42)
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    return train_sampler, val_sampler


train_sampler, val_sampler = dataset_sampler(data_all)

batch_size = 20
trainloader = data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=train_sampler)
validloader = data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=val_sampler)

testloader = data.DataLoader(test_data, batch_size=batch_size, num_workers=0)
##获得一个batch的数据
for step, (b_x, b_y) in enumerate(trainloader):
    if step > 0:
        break

print(b_x.shape)
print(b_y.shape)
print("图像的取值范围为：", b_x.min(), "~", b_x.max())
print(classes)
# %%
import matplotlib.pyplot as plt
import numpy as np


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


inputs, classes = next(iter(trainloader))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out)

# %%
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = models.vgg16(pretrained=True)

# %%
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import torch.optim as optim

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# %%

num_epochs = 10
num_classes = 2
batch_size = 25
learning_rate = 0.001

epochs = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device
# %%
train_losses = []
valid_losses = []
for epoch in range(1, num_epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0
    net.train()
    for data, label in trainloader:
        data = data.to(device)
        target = label.to(device)
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = net(data)
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)
        # validate-the-model
        net.eval()
    for data, target in validloader:
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        loss = criterion(output, target)
        # update-average-validation-loss
        valid_loss += loss.item() * data.size(0)

    # calculate-average-losses
    train_loss = train_loss / len(trainloader.sampler)
    valid_loss = valid_loss / len(validloader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # print-training/validation-statistics
    print('Epoch: {} /tTraining Loss: {:.6f} /tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

print('Finish training')

# %%
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# %%
# test
test_dataiter = iter(testloader)

# print the image
image, labels = test_dataiter.next()
imshow(torchvision.utils.make_grid(image))
# %%
net = Net()
net.load_state_dict(torch.load(PATH))

# %%
outputs = net(image)

# %%
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# %%
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

# %%
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
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
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))

# %%

plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
