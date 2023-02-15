# -*- coding: utf-8 -*-
"""Ruthwik_part_3_flatGenrerpart_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u8t2rOc0Xwa1_5mPqQTYbpTY1IRrQ2dy
"""

pip install tensorflow==2.4

import tensorflow as tf
import numpy as np
import torch
import torchvision as tv
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# Grab MNIST dataset
trainingSet = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
testingSet = datasets.MNIST('', train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))

# Set different batch sizes for each of the 5 different models
batchSizes = [10, 30, 120, 500, 800]
train1 = torch.utils.data.DataLoader(trainingSet, batch_size=batchSizes[0], shuffle=True)
test1 = torch.utils.data.DataLoader(testingSet, batch_size=batchSizes[0], shuffle=True)
train2 = torch.utils.data.DataLoader(trainingSet, batch_size=batchSizes[1], shuffle=True)
test2 = torch.utils.data.DataLoader(testingSet, batch_size=batchSizes[1], shuffle=True)
train3 = torch.utils.data.DataLoader(trainingSet, batch_size=batchSizes[2], shuffle=True)
test3 = torch.utils.data.DataLoader(testingSet, batch_size=batchSizes[2], shuffle=True)
train4 = torch.utils.data.DataLoader(trainingSet, batch_size=batchSizes[3], shuffle=True)
test4 = torch.utils.data.DataLoader(testingSet, batch_size=batchSizes[3], shuffle=True)
train5 = torch.utils.data.DataLoader(trainingSet, batch_size=batchSizes[4], shuffle=True)
test5 = torch.utils.data.DataLoader(testingSet, batch_size=batchSizes[4], shuffle=True)

# Calculate the number of parameters in a neural network
def calcParams(inputModel):
    val = sum(params.numel() for params in inputModel.parameters() if params.requires_grad)
    return val

# Model 1 - 2 Hidden layer / 20715 Parameters
class Model1 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val

# Model 2 - 2 Hidden layer / 20715 Parameters
class Model2 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val

# Model 3 - 2 Hidden layer / 20715 Parameters
class Model3 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val
    
# Model 4 - 2 Hidden layer / 20715 Parameters
class Model4 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val
    
# Model 5 - 2 Hidden layer / 20715 Parameters
class Model5 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val

model1 = Model1()
model2 = Model2()
model3 = Model3()
model4 = Model4()
model5 = Model5()
print(calcParams(model1))
print(calcParams(model2))
print(calcParams(model3))
print(calcParams(model4))
print(calcParams(model5))

# Set up necessary auxilaries for neural net training
model1 = Model1()
model2 = Model2()
model3 = Model3()
model4 = Model4()
model5 = Model5()
costFunc = nn.CrossEntropyLoss()
model1Opt = optim.Adam(model1.parameters(), lr=0.001)
model2Opt = optim.Adam(model2.parameters(), lr=0.001)
model3Opt = optim.Adam(model3.parameters(), lr=0.001)
model4Opt = optim.Adam(model4.parameters(), lr=0.001)
model5Opt = optim.Adam(model5.parameters(), lr=0.001)

# Train all 5 models using different batch sizes
EPOCHS = 50
for index in range(EPOCHS):
    # Print index to show progress of training during computation
    print(index)
    
    # Model 1
    for batch in train1:
        inputImages, groundTruth = batch
        model1.zero_grad()
        output = model1(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model1Opt.step()

    # Model 2
    for batch in train2:
        inputImages, groundTruth = batch
        model2.zero_grad()
        output = model2(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model2Opt.step()
        
    # Model 3
    for batch in train3:
        inputImages, groundTruth = batch
        model3.zero_grad()
        output = model3(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model3Opt.step()
        
    # Model 4
    for batch in train4:
        inputImages, groundTruth = batch
        model4.zero_grad()
        output = model4(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model4Opt.step()
        
    # Model 5
    for batch in train5:
        inputImages, groundTruth = batch
        model5.zero_grad()
        output = model5(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model5Opt.step()

# Calculate loss and accuracy for every model for training set
trainCostList = []
trainAccList = []

# Model 1
correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train1:
        inputImages, groundTruth = batch
        output = model1(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
trainCostList.append(costTotal / costCounter)
trainAccList.append(round(correct/total, 3)) 

# Model 2
correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train2:
        inputImages, groundTruth = batch
        output = model2(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
trainCostList.append(costTotal / costCounter)
trainAccList.append(round(correct/total, 3)) 

# Model 3
correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train3:
        inputImages, groundTruth = batch
        output = model3(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
trainCostList.append(costTotal / costCounter)
trainAccList.append(round(correct/total, 3)) 

# Model 4
correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train4:
        inputImages, groundTruth = batch
        output = model4(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
trainCostList.append(costTotal / costCounter)
trainAccList.append(round(correct/total, 3)) 

# Model 5
correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train5:
        inputImages, groundTruth = batch
        output = model5(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
trainCostList.append(costTotal / costCounter)
trainAccList.append(round(correct/total, 3))

# Calculate loss and accuracy for every model for testing set
testCostList = []
testAccList = []

# Model 1
correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test1:
        inputImages, groundTruth = batch
        output = model1(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
testCostList.append(costTotal / costCounter)
testAccList.append(round(correct/total, 3))

# Model 2
correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test2:
        inputImages, groundTruth = batch
        output = model2(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
testCostList.append(costTotal / costCounter)
testAccList.append(round(correct/total, 3))

# Model 3
correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test3:
        inputImages, groundTruth = batch
        output = model3(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
testCostList.append(costTotal / costCounter)
testAccList.append(round(correct/total, 3))

# Model 4
correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test4:
        inputImages, groundTruth = batch
        output = model4(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
testCostList.append(costTotal / costCounter)
testAccList.append(round(correct/total, 3))

# Model 5
correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test5:
        inputImages, groundTruth = batch
        output = model5(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
testCostList.append(costTotal / costCounter)
testAccList.append(round(correct/total, 3))

# Calculate sensitivity of every model
sensitivityList = []

# Model 1
# Get gradient norm (From slides)
gradAll = 0.0
fNormAll = 0
counter = 0
for p in model1.parameters():
    grad = 0.0
    if p.grad is not None:
        grad = p.grad
        # Calculate Frobenius norm of gradients
        fNorm = torch.linalg.norm(grad).numpy()
        fNormAll += fNorm
        counter += 1
sensitivityList.append(fNormAll / counter)


# Model 2
gradAll = 0.0
fNormAll = 0
counter = 0
for p in model2.parameters():
    grad = 0.0
    if p.grad is not None:
        grad = p.grad
        fNorm = torch.linalg.norm(grad).numpy()
        fNormAll += fNorm
        counter += 1
sensitivityList.append(fNormAll / counter)

# Model 3
gradAll = 0.0
fNormAll = 0
counter = 0
for p in model3.parameters():
    grad = 0.0
    if p.grad is not None:
        grad = p.grad
        fNorm = torch.linalg.norm(grad).numpy()
        fNormAll += fNorm
        counter += 1
sensitivityList.append(fNormAll / counter)

# Model 4
gradAll = 0.0
fNormAll = 0
counter = 0
for p in model4.parameters():
    grad = 0.0
    if p.grad is not None:
        grad = p.grad
        fNorm = torch.linalg.norm(grad).numpy()
        fNormAll += fNorm
        counter += 1
sensitivityList.append(fNormAll / counter)

# Model 5
gradAll = 0.0
fNormAll = 0
counter = 0
for p in model5.parameters():
    grad = 0.0
    if p.grad is not None:
        grad = p.grad
        fNorm = torch.linalg.norm(grad).numpy()
        fNormAll += fNorm
        counter += 1
sensitivityList.append(fNormAll / counter)

# Visulaize Accuracy and Sensitivity by batch size of model
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(batchSizes, trainAccList, 'g--', label='Train')
ax1.plot(batchSizes, testAccList, 'c', label='Test')
ax2.plot(batchSizes, sensitivityList, 'b', label='Sensitivity')
ax1.set_title('Effect of Batch Size on Accuracy and Sensitivity')
ax1.set_xlabel('Batch Sizes')
ax1.set_xscale('log')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Sensitivity')
ax1.legend(loc='upper right')

# Visulaize Loss and Sensitivity by batch size of model
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(batchSizes, trainCostList, 'g--', label='Train')
ax1.plot(batchSizes, testCostList, 'c', label='Test')
ax2.plot(batchSizes, sensitivityList, 'b', label='Sensitivity')
ax1.set_title('Effect of Batch Size on Loss and Sensitivity')
ax1.set_xlabel('Batch Sizes')
ax1.set_xscale('log')
ax1.set_ylabel('Loss')
#colo='b'
ax2.set_ylabel('Sensitivity')
#color='r'
ax1.legend(loc='upper right')
