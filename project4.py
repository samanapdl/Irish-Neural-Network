# -*- coding: utf-8 -*-
"""

@author: sammy
"""
from sklearn.datasets import load_iris
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


#load dataset
iris_data = load_iris()
features, pre_labels = iris_data.data, iris_data.target

labels = []
for num in range(len(pre_labels)):
    if pre_labels[num] == 0:
        labels.append([1, 0, 0])
    if pre_labels[num] == 1:
        labels.append([0, 1, 0])
    if pre_labels[num] == 2:
        labels.append([0, 0, 1])

labels = np.array(labels, dtype = float)

feature_train, feature_test, labels_train, labels_test = train_test_split(features, labels, test_size= 0.3)

feature_train_v = Variable(torch.FloatTensor(feature_train), requires_grad = False)
labels_train_v = Variable(torch.FloatTensor(labels_train), requires_grad = False)
feature_test_v = Variable(torch.FloatTensor(feature_test), requires_grad = False)
labels_test_v = Variable(torch.FloatTensor(labels_test), requires_grad = False)

#print(feature_test_v)

class Net(nn.Module):
      def __init__(self):
        super(Net, self).__init__()
        self.h0 = nn.Linear(4,10)
        self.out = nn.Linear(10,3)

      def forward(self, x):
        x = self.h0(x)
        x = F.tanh(x)
        x = self.out(x)
        x = F.tanh(x)
        return x
ANN = Net()

def _accuracy(targets,predict):
    return (targets.argmax(1)==predict).sum().item()/predict.size()[0]*100

training_loss=[]
for a in np.arange(0.01,0.11,0.01):
    for m in np.arange (0.5,1,0.2):
        ANN = Net()
        optimizer = torch.optim.SGD(ANN.parameters(), lr=a, momentum= m)
        loss_fn= nn.MSELoss()
        for epoch in range(50):
            optimizer.zero_grad()
            out = ANN(feature_train_v)
            loss_p = loss_fn(out,labels_train_v)
            loss_p.backward()
            optimizer.step()
            training_loss.append(loss_p)
          
        mean_loss= sum(training_loss) / len(training_loss)
        print ("lr ",a," momentum: ",m,"loss", mean_loss.data)

print("The best value for the learning rate and momentum we obtained are: 0.1 and 0.9")
                 

#Training 
test_losses=[] 
train_losses=[]
train_acc=[]
test_acc=[]
ANN = Net()
optimizer = torch.optim.SGD(ANN.parameters(), lr=0.1, momentum= 0.9)
loss_fn= nn.MSELoss()
predicted_values=[]

for epoch in range(500):
    optimizer.zero_grad()
    out = ANN(feature_train_v)
    loss = loss_fn(out,labels_train_v)
    train_predict=out.argmax(1)
    acc=_accuracy(labels_train_v,train_predict)
    train_losses.append(loss)
    train_acc.append(acc)
    loss.backward()
    optimizer.step() 
    
  
    test_out=ANN(feature_test_v)
    loss1 = loss_fn(test_out,labels_test_v)
    test_losses.append(loss1)
    test_predict=test_out.argmax(1)
    acc1=_accuracy(labels_test_v,test_predict)
    test_acc.append(acc1)

        
##Graph Plot
import matplotlib.pyplot as plt
train_losses = np.array(train_losses, dtype = np.float)
test_losses = np.array(test_losses, dtype = np.float)
plt.plot(train_losses,color='red', label='train')
plt.plot(test_losses,color='blue', label='test')
plt.title('Losses Plot')
plt.legend()
plt.show()

plt.title('Accuracy Plot')
plt.plot(train_acc,color="red",label='train')
plt.plot(test_acc,color="blue",label='test')
plt.legend()
plt.show()

#############################################################################
#Validating the model

predicted_values = []
for num in range(len(feature_train_v)):
    predicted_values.append(ANN(feature_train_v[num]))
score = 0
for num in range(len(predicted_values)):
    if np.argmax(labels_train[num]) == np.argmax(predicted_values[num].data.numpy()):
        score = score + 1
accuracy = float(score / len(predicted_values)) * 100
print ('FInal Training Accuracy Score is ' + str(accuracy))
#############################################################################

predicted_values1 = []
for num in range(len(feature_test_v)):
    predicted_values1.append(ANN(feature_test_v[num]))
score = 0
for num in range(len(predicted_values1)):
    if np.argmax(labels_test[num]) == np.argmax(predicted_values1[num].data.numpy()):
        score = score + 1
accuracy = float(score / len(predicted_values1)) * 100
print ('Final Testing Accuracy Score is ' + str(accuracy))
