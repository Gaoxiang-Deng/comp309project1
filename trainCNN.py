# %%
import Augmentor


def aungmentPipeline(dir):
    p = Augmentor.Pipeline(dir)
    p.rotate(probability=0.7, max_right_rotation=25, max_left_rotation=25)
    p.zoom(probability=0.3, min_factor=1.1, max_factor=1.6)
    p.skew(.1)
    p.flip_left_right(.5)
    p.crop_centre(.1, .2, randomise_percentage_area=True)
    p.sample(5000)


# %%
aungmentPipeline("C:/workspace/traindata/traindata/cherry")
aungmentPipeline("C:/workspace/traindata/traindata/strawberry")
aungmentPipeline("C:/workspace/traindata/traindata/tomato")


# %%
import matplotlib
import torch
import torchvision
from ipykernel.pylab.config import InlineBackend
from jedi.api.refactoring import inline

from torchvision import transforms


transform = transforms.Compose(
    [transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.Resize([32, 32]),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose(
    [transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.Resize([32, 32]),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

classes = ('cherry', 'strawberry', 'tomato')

data_path = "C:/workspace/comp309project/data/traindata"
data_all = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
test_data_path = "C:/workspace/comp309project/data/testdata"
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


# %%
import matplotlib.pyplot as plt
import numpy as np


def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


# %%
dataiter = iter(trainloader)
images, labels = dataiter.next()

fig, axes = plt.subplots(figsize=(12, 12), ncols=5)
print('training images')
for i in range(5):
    axe1 = axes[i]
    imshow(torchvision.utils.make_grid(images))

print(images[0].size())
print(','.join('%5s' % classes[labels[j]] for j in range(batch_size)))

# %%
num_epochs = 75
num_classes = 2
batch_size = 25
learning_rate = 0.001

epochs = 75


# %%
# Model 1, 4 lay of cnn
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=5)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 2 * 2, 50)
        self.fc2 = nn.Linear(50, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

# %%
# VGG16 model
import torch.nn.functional as F
from torchvision import models
from torch.nn import Linear, Sequential

net = models.vgg16_bn(pretrained=True)
for param in net.parameters():
    param.requires_grad = False

if torch.cuda.is_available():
    model = net.cuda()

# Add on classifier
net.classifier[6] = Sequential(Linear(4096, 3))

for param in net.classifier[6].parameters():
    param.requires_grad = True

# %%
# 2 lay cnn model
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# %%

net = Net()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)
# %%

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
        target = nn.functional.one_hot(target % 3, num_classes=3)
        target = target.float()
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
        target = nn.functional.one_hot(target % 3, num_classes=3)
        target = target.float()
        loss = criterion(output, target)
        # update-average-validation-loss
        valid_loss += loss.item() * data.size(0)

    # calculate-average-losses
    train_loss = train_loss / len(trainloader.sampler)
    valid_loss = valid_loss / len(validloader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # print-training/validation-statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

print('Finish training')


# %%
PATH = './cnn1_net.pth'
# torch.save(net.state_dict(), PATH)

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

