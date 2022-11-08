import pandas as pd
from sklearn.metrics import classification_report

import torch
from torchvision import transforms
from torchvision import models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn

from PIL import Image

torch.manual_seed(42)

import torch.optim as optim

train_df = pd.read_csv("train_label.csv")
print(train_df, len(train_df) )

root = "/content/drive/My Drive/Colab Notebooks/cs5242-data/train_image/train_image/"
img_path = train_df['ID'][0]

img = Image.open(root + str(img_path) + '.png' ) #similar to cv2.imread()
img

class CS5242_dataset(Dataset): 
    
    def __init__(self, root_path , dataframe, transform=None):
        
        self.df = dataframe    
        self.transform = transform
        self.root_path = root_path
        
        self.image_paths = self.df['ID'] #image names
        self.labels = self.df['Label']
                

    def __getitem__(self, index):
        
        img_path = self.image_paths[index] 
        image = Image.open(self.root_path + str(img_path) + '.png')
        
        target = torch.tensor(self.labels[index])
      
        if self.transform != None:
            image = self.transform(image)
          
        return [image, target]
       
    def __len__(self):
        return len(self.df)
    
    
batch_size = 256
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transformations = transforms.Compose([
                transforms.Resize(size=(224,224),interpolation=2),
                transforms.ToTensor(), #3*H*W, [0, 1]
                normalize]) # normalize with mean/std


train_dataset = CS5242_dataset(root, train_df, transform = transformations)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2, shuffle = True, pin_memory=True)
train_dataset.__getitem__(0)

class Network(nn.Module):
    def __init__(self, pretrained = True, num_classes = 3, drop_rate = 0):
        super(Network, self).__init__()
        resnet = models.resnet18(pretrained) #https://pytorch.org/vision/0.8/models.html
        
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512
        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x8

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)

        return F.softmax(x, dim=1) #classification output
    
model = Network() #instantiate
model.to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
print(model)


num_epochs = 10
criterion = nn.CrossEntropyLoss()

num_correct = 0
num_samples = 0

model.train() #set model in training mode # if testing model, use model.eval()
for epoch in range(num_epochs):
    for (x, y) in train_loader:
      
        x = x.to("cuda") #images
        y = y.to("cuda") #unpacks labels

        preds = model(x) #forward pass

        loss = criterion(preds, y)

        optimizer.zero_grad() # backward
        loss.backward()
        optimizer.step() # gradient descent or adam step

        num_correct += torch.sum(torch.eq(preds.argmax(1), y)).item()
        num_samples += preds.size(0)



    train_accuracy = num_correct/num_samples
    print(epoch, train_accuracy)

preds.sum(1)

def load_transform(image_size=256, crop_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.Resize(image_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.5),
        # AddGaussianNoise(0., 0.001),
        transforms.Normalize(mean=mean, std=std)
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size=(image_size, image_size), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train_transform, valid_transform
