import torch device = "cpu"
class LSTMVAE(torch.nn.Module):
def   init  (self):
super().  init  ()
self.lstm = torch.nn.LSTM(36,32, batch_first = True) self.mu = torch.nn.Linear(32,16)
self.mid = torch.nn.Linear(32,20) self.mid2mu = torch.nn.Linear(20,16) self.logsigma = torch.nn.Linear(32,16)
self.lstm2 = torch.nn.LSTM(16,32, batch_first = True) self.fc = torch.nn.Linear(32,36)
self.relu = torch.nn.ReLU() self.kl = 0

def forward(self,x):
x,_ = self.lstm(x)
x = x[:,-1,:] # output of the last value of the sequence => last hidden value x = self.relu(x)

mu_temp = self.mid(x)
mu = self.mid2mu(mu_temp) logsigma = torch.exp(self.logsigma(x))
z = mu + torch.randn(mu.shape).to(device) * torch.exp(0.5 * logsigma ) x = z.unsqueeze(1)
x = x.repeat(1,4,1) ## make sure to add sequence size here x,_ = self.lstm2(x)
x = self.fc(x)
return x, mu, logsigma
 
Autoencoder 1
This is the first of the three autoencoders used in the ensemble method
class encoder(nn.Module):

def  init (self, inputSize): super(). init ()

self.fc1 = nn.Linear(inputSize, 2*inputSize) self.fc2 = nn.Linear(2*inputSize, int(inputSize/2))
self.fc3 = nn.Linear(int(inputSize/2), int(inputSize/2))

def forward(self, x):
x = F.relu(self.fc1(x)) x = F.relu(self.fc2(x))
x = F.relu(self.fc3(x) ) # No relu on the last layer return x

class decoder(nn.Module):

def  init (self, inputSize): super(). init ()
self.fc1 = nn.Linear(int(inputSize/2), 2*inputSize) self.fc2 = nn.Linear(2*inputSize, inputSize)

def forward(self, x):
x = F.relu(self.fc1(x))
x = self.fc2(x) # No relu on the last layer to reconstruct the
input
return x



Autoencoder 2
This is the second of the three autoencoders used in the ensemble method
class encoder2(nn.Module):

def  init (self, inputSize): super(). init ()

self.fc1 = nn.Linear(inputSize, 2*inputSize) self.fc2 = nn.Linear(2*inputSize, int(inputSize/2))
self.fc3 = nn.Linear(int(inputSize/2), int(inputSize/2)) self.fc4 = nn.Linear(int(inputSize/2), int(inputSize/4))
 
def forward(self, x):
x = F.relu(self.fc1(x)) x = F.relu(self.fc2(x)) x = F.relu(self.fc3(x) ) x = F.relu(self.fc4(x) ) return x

class decoder2(nn.Module):

def  init (self, inputSize): super(). init ()
self.fc1 = nn.Linear(int(inputSize/4), int(inputSize/2)) self.fc2 = nn.Linear(int(inputSize/2), inputSize) self.fc3 = nn.Linear(inputSize, inputSize)

def forward(self, x):
x = F.relu(self.fc1(x)) x = F.relu(self.fc2(x))
x = self.fc3(x) # No relu on the last layer to reconstruct the
input
return x


Autoencoder 3
This is the third of the three autoencoders used in the ensemble method
class encoder3(nn.Module):

def  init (self, inputSize): super(). init ()

self.fc1 = nn.Linear(inputSize, inputSize) self.fc2 = nn.Linear(inputSize, inputSize) self.fc3 = nn.Linear(inputSize, int(inputSize/2))
self.fc4 = nn.Linear(int(inputSize/2), int(inputSize/2)) self.fc5 = nn.Linear(int(inputSize/2), int(inputSize/6))

def forward(self, x):
x = F.relu(self.fc1(x)) x = F.relu(self.fc2(x)) x = F.relu(self.fc3(x) )
# x = F.relu(self.fc4(x) ) # x = F.relu(self.fc5(x) ) return x
 
class decoder3(nn.Module):

def  init (self, inputSize): super(). init ()
# self.fc1 = nn.Linear(int(inputSize/6), int(inputSize/2)) # self.fc2 = nn.Linear(int(inputSize/2), int(inputSize/2)) self.fc3 = nn.Linear(int(inputSize/2), inputSize)
self.fc4 = nn.Linear(inputSize, inputSize) self.fc5 = nn.Linear(inputSize, inputSize)

def forward(self, x):
# x = F.relu(self.fc1(x)) # x = F.relu(self.fc2(x)) x = F.relu(self.fc3(x)) x = F.relu(self.fc4(x))
x = self.fc5(x) # No relu on the last layer to reconstruct the
input
return x




Frontend


import torch
from torchvision import models
import torchvision.transforms as transforms from PIL import Image
from flask import Flask, jsonify, request from myLstm import LSTMVAE

app = Flask(  name  )

device = torch.device("cpu") iotModel = LSTMVAE()
iotModel.load_state_dict(torch.load("stateDict.pth", map_location = device)) iotModel.eval()
loss_fn = torch.nn.MSELoss().to(device)

@app.route('/predict', methods=['POST']) def predict():
if request.method == 'POST': request_data = request.get_json()
dataTensor = torch.tensor(request_data['val'])
 
input = torch.unsqueeze(dataTensor, dim = 0) out,_,_ = iotModel(input)
loss = loss_fn(out, input) return str(loss.item())


@app.route('/') def hello():
return 'Hello World!'

if  name  == ' main ': app.run()



Backend
import pandas as pd import requests import streamlit as st import json
import torch
import numpy as np import time

st.set_page_config(
page_title='SWaT Testbed Monitoring', layout="wide",
)


@st.cache
def readData():
data = pd.read_excel("smallNoDate.xlsx")
# data = pd.read_excel("microNoDate.xlsx") return data
df = readData()

meannDev = pd.read_excel("meanNdev.xlsx") means = meannDev.loc[0].values.flatten().tolist() devs = meannDev.loc[1].values.flatten().tolist()


left_column, right_column,third = st.columns(3) # You can use a column just like st.sidebar:
 
left_column.subheader("FIT101") A = left_column.line_chart() left_column.subheader("LIT101") B = left_column.line_chart()

with right_column: right_column.subheader("MV101") C = st.line_chart() right_column.subheader("P101") D = st.line_chart()

third.subheader("ML Model error") E = third.line_chart()
third.subheader("Attack (1) or Not (0) ") F = third.line_chart()

windowSize = 1 data = []
for item in df.itertuples(index=False, name=None):

val = list(item)
# Standardize the data
item = val[:-1] # remove Attack/Normal
subtracted = [element1 - element2 for (element1, element2) in zip(item, means)] divided = [element1/element2 for (element1, element2) in zip(subtracted, devs)]

data.append(divided) A.add_rows([divided[0]]) B.add_rows([divided[1]]) C.add_rows([divided[2]]) D.add_rows([divided[3]]) F.add_rows([val[-1]])

if windowSize % 4 == 0:
jsonData = {"val": data }
resp = requests.post("http://localhost:5000/predict", json = jsonData)
retVal = float(resp.text)
E.add_rows([[retVal],[retVal],[retVal],[retVal]]) # one for each window value as we slide by window size windowSize = 0
data = [] resp.text
time.sleep(0.01) windowSize += 1
Code for model training.
 
Similar code can be used to train LSTM VAEs, autoencoders and other unsupervised models. Tweaks in the model architecture can lead to big changes in the overall model performance.

%matplotlib inline import torch
import torch.nn as nn
import torch.nn.functional as F import torch.optim as optim
import pandas as pd ## ideally remove this and build dataloader directly from CSV file and applying transforms on the way
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler import torch.utils.data as data_utils
import torch.optim as optim
from sklearn.metrics import confusion_matrix import seaborn as sns
import numpy as np
from torch.utils.data import Dataset from torch.utils.data import DataLoader import pandas as pd
import random as rand
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_excel("SWaT_Dataset_Attack_mod.xlsx") df_backup = df

df.rename(columns={'Normal/Attack':'y'},inplace=True) df['y'] = df['y'].replace(['Normal'],0)
df['y'] = df['y'].replace(['Attack'],1)

##updating the wrongly spelt cells and then checking df['y'] = df['y'].replace(['A ttack'],1)
df.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())

#Seeing the distribution of "y" labels df['y'].plot.hist(bins = 200,fc=(0, 0, 1, 0.2));

SEED = 123
frames_training = [df[30000:60000], df[395000:430000]] df_train = pd.concat(frames_training)
 
frames = [df[:30000], df[60000:395000],df[430000:]]
df_reduced = pd.concat(frames)

df_test, df_valid = train_test_split(df_reduced, test_size=0.2, random_state=SEED)

df_valid = df_valid[df_valid['y'] ==0] ## This will also give us poor results as ideally even the validation set should be selected from clean normal data

###Standardizing train test and validation ##https://intellipaat.com/community/4548/apply-standardscaler-on-a- partial-part-of-a-data-set
###The names of the columns have been kept based on the erroneous names in excel sheet. Random spaces exist in names and are to be kept as such
cols = [' MV101','P101','P102',' MV201',' P201',' P202','P203','
P204','P205','P206','MV301','MV302',' MV303','MV304','P301','P302'
,'P401','P402','P403','P404','UV401','P501','P502','P601','P602','P603','y
',' Timestamp']

###Training
toStandardize = df_train.loc[:, ~df_train.columns.isin(cols)] ct = ColumnTransformer([('remark', StandardScaler(),
toStandardize.columns.values.tolist())], remainder='passthrough') df_train_noTime = df_train.loc[:, ~df_train.columns.isin([' Timestamp','y'])]
df_train_standard = ct.fit_transform(df_train_noTime)

###Testing
toStandardize = df_test.loc[:, ~df_test.columns.isin(cols)] ct = ColumnTransformer([('remark', StandardScaler(),
toStandardize.columns.values.tolist())], remainder='passthrough') df_test_noTime = df_test.loc[:, ~df_test.columns.isin([' Timestamp','y'])] df_test_standard = ct.fit_transform(df_test_noTime)

###Validation
toStandardize = df_valid.loc[:, ~df_valid.columns.isin(cols)] ct = ColumnTransformer([('remark', StandardScaler(),
toStandardize.columns.values.tolist())], remainder='passthrough') df_valid_noTime = df_valid.loc[:, ~df_valid.columns.isin([' Timestamp','y'])]
df_valid_standard = ct.fit_transform(df_valid_noTime)

class DataSetMy(Dataset):
def  init (self,panda,window): self.x = panda
self.window = window def  len (self):
return len(self.x)-self.window+1 def  getitem (self,idx):
x = self.x[idx:idx+self.window] y = self.x[idx:idx+self.window] return x, y


class DataSimple(Dataset):
def  init (self,panda,window): self.x = panda
self.window = window def  len (self):
return len(self.x)
def  getitem (self,idx): x = self.x[idx]
return x, x



##Converting standardized pandas DFs to torch tensors. These will be converted to windowed datasets and
## subsequently datloaders batch_size = 128
#training
train = torch.tensor(df_train_standard.astype('float32')) #validation
valid = torch.tensor(df_valid_standard.astype('float32'))



# generating windowed dataset and a dataloader train_tensor = DataSetMy(train,4)
train_loader = DataLoader(train_tensor, batch_size =128)

train_tensor_simple = DataSimple(train,4)
train_loader_simple = DataLoader(train_tensor_simple, batch_size =128)



class LSTMAutoencoder(torch.nn.Module):

def  init (self): super(). init ()
# self.lstm = torch.nn.LSTM(51,30, batch_first = True) self.fc1 = torch.nn.Linear(51,30)
self.fc2 = torch.nn.Linear(30,15) self.fc3 = torch.nn.Linear(15,30) self.fc4 = torch.nn.Linear(30,51) self.relu = torch.nn.ReLU()
# self.lstm2 = torch.nn.LSTM(30,51, batch_first = True)

def forward(self,x):
x = self.relu(self.fc1(x)) x = self.relu(self.fc2(x)) x = self.relu(self.fc3(x)) x = self.fc4(x)
return x

class LSTM_AE_2(torch.nn.Module): def  init (self):
super(). init ()
self.lstm = torch.nn.LSTM(51,30, batch_first = True) self.fc1 = torch.nn.Linear(30,15)
self.relu = torch.nn.ReLU() self.fc2 = torch.nn.Linear(15,30)
self.lstm2 = torch.nn.LSTM(30,51, batch_first = True)

def forward(self,x): x,_ = self.lstm(x) x = x[:,-1,:]
x = self.fc1(x) x = self.relu(x) x = self.fc2(x)
x = x.unsqueeze(1) x = x.repeat(1,4,1) x,_ = self.lstm2(x) return x


class LSTMVAE(torch.nn.Module): def  init (self):
super(). init ()
self.lstm = torch.nn.LSTM(51,32, batch_first = True) self.mu = torch.nn.Linear(32,16)
self.logsigma = torch.nn.Linear(32,16)
self.lstm2 = torch.nn.LSTM(16,32, batch_first = True)

def  init (self): super(). init ()
# self.lstm = torch.nn.LSTM(51,30, batch_first = True) self.fc1 = torch.nn.Linear(51,30)
self.fc2 = torch.nn.Linear(30,15) self.fc3 = torch.nn.Linear(15,30) self.fc4 = torch.nn.Linear(30,51) self.relu = torch.nn.ReLU()
# self.lstm2 = torch.nn.LSTM(30,51, batch_first = True)

def forward(self,x):
x = self.relu(self.fc1(x)) x = self.relu(self.fc2(x)) x = self.relu(self.fc3(x)) x = self.fc4(x)
return x

class LSTM_AE_2(torch.nn.Module): def  init (self):
super(). init ()
self.lstm = torch.nn.LSTM(51,30, batch_first = True) self.fc1 = torch.nn.Linear(30,15)
self.relu = torch.nn.ReLU() self.fc2 = torch.nn.Linear(15,30)
self.lstm2 = torch.nn.LSTM(30,51, batch_first = True)

def forward(self,x): x,_ = self.lstm(x) x = x[:,-1,:]
x = self.fc1(x) x = self.relu(x) x = self.fc2(x)
x = x.unsqueeze(1) x = x.repeat(1,4,1) x,_ = self.lstm2(x) return x


class LSTMVAE(torch.nn.Module): def  init (self):
super(). init ()
self.lstm = torch.nn.LSTM(51,32, batch_first = True) self.mu = torch.nn.Linear(32,16)
self.logsigma = torch.nn.Linear(32,16)
self.lstm2 = torch.nn.LSTM(16,32, batch_first = True)

self.fc = torch.nn.Linear(32,51) self.relu = torch.nn.ReLU() self.kl = 0

def forward(self,x): x,_ = self.lstm(x)
x = x[:,-1,:] # just get the output of the last value of the sequence. basically the last hidden value
mu = self.mu(x)
logsigma = torch.exp(self.logsigma(x))
z = mu + torch.randn(mu.shape).to(device) * torch.exp(0.5 * logsigma ) x = z.unsqueeze(1)
x = x.repeat(1,4,1) ## make sure to add sequence size here x,_ = self.lstm2(x)
x = self.fc(x)
## Note: Can't use ReLU here as this has to be the same as the input.
Although, ReLU gives back only +ve values, and all our values are positive. So it should work
return x, mu, logsigma



h = LSTMVAE()
pytorch_total_params = sum(p.numel() for p in h.parameters()) print("total parameters in LSTMVAE: ", pytorch_total_params)

h = LSTMAutoencoder()
pytorch_total_params = sum(p.numel() for p in h.parameters()) print("total parameters in LSTM AE: ", pytorch_total_params)
# Training epoch = 100
netLSTM_AE = LSTMAutoencoder().to(device)
optimizer = torch.optim.Adam(netLSTM_AE.parameters(),0.0001) netLSTM_AE.train()

for i in range(epoch): count = 0
loss_fn = torch.nn.MSELoss().to(device) for x,y in train_loader_simple:
x,y = x.to(device), y.to(device) count = count + 1 optimizer.zero_grad()
out = netLSTM_AE(x)

loss = loss_fn(out,y) loss.backward() optimizer.step()
print("epoch: ", i, " loss : ", loss.item())



epoch = 20
net = LSTMVAE().to(device)
optimizer = torch.optim.Adam(net.parameters(),0.01) kl_weight = 0.0005
net.train()

for i in range(epoch): count = 0
loss_fn = torch.nn.MSELoss().to(device) for x,y in train_loader:
x,y = x.to(device), y.to(device) count = count + 1 optimizer.zero_grad() out,mu,logsigma = net(x)

#calculating the loss
latent_loss = 0.5 * torch.sum(torch.exp(logsigma) + mu**2 - 1.0 - logsigma)
reconstruction_loss = loss_fn(out,y)
loss = kl_weight * latent_loss + reconstruction_loss loss.backward()
optimizer.step() print("loss : ", loss.item())

THRESHOLD = .5

#craeting windowed test inputs to generate windowed classification the test set
testtFormat = torch.tensor(df_test_standard.astype('float32')) test_tensor = DataSetMy(testtFormat,4)
test_loader = DataLoader(test_tensor, batch_size =1)



#calculating the losses for regeneration of all the windows... test_loss =[]


count = 0
for x,y in test_loader: count +=1
x,y = x.to(device), y.to(device) out,_,_ = net(x)
loss = loss_fn(out,y) test_loss.append(loss.item())
del x # to make sure RAM usage stays within limit del y
if count % 10000 == 0:
print (" loops over: ", count // 10000)



#creating an error database for confusion matrix calculations error_df_test = pd.DataFrame({'Reconstruction_error': test_loss, 'Timestamp' : df_test[' Timestamp'][:len(test_loss)],
'True_class': df_test['y'][:len(test_loss)]})

error_df_test['predicted_class'] = 0
condition = error_df_test.Reconstruction_error > 2*THRESHOLD ##This value is esentially the validation loss. basically on unseen normal data, the loss can be as high as this value everythin else (attack) should be much higher
error_df_test.loc[condition, "predicted_class"] = 1 error_df_test['predicted_class'].sum()

#conf matrix
conf_matrix = confusion_matrix(error_df_test.True_class, error_df_test.predicted_class)
plt.figure(figsize=(8, 8)) LABELS = ["Normal","Attack"]
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix") plt.ylabel('True class') plt.xlabel('Predicted class') plt.show() 
