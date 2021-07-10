import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import DogDataset
from model import Tesseract
from EarlyStop import EarlyStopping
import torchvision
from torch.nn import functional as F
import albumentations as A
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)
     
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
p=0.5
    
class CatDogModel(Tesseract):
     def __init__(self, num_class, pretrained=False):
         super().__init__() 
         self.backbone = torchvision.models.resnet18(pretrained=pretrained)    
         in_features = self.backbone.fc.in_features
         self.out = nn.Linear(in_features, num_class)
         
     def loss_fn(self, outputs, targets):
         loss = nn.CrossEntropyLoss()(outputs, targets)
         return loss
     
     def metrics_fn(self, outputs, targets):
         outputs = torch.argmax(outputs, axis=1).cpu().detach().numpy()
         targets = targets.cpu().detach().numpy()
         acc = accuracy_score(targets, outputs)
         return {'accuracy_score': acc}
     
     def fetch_optimizer(self):
         opt = torch.optim.Adam(self.parameters(), lr=1e-4)
         return opt
     
     def fetch_scheduler(self):
         sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
             self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
         )
         return sch
         
     def forward(self, image, targets=None):
         batch_size, C, H, W = image.shape
         x = self.backbone.conv1(image)
         x = self.backbone.bn1(x)
         x = self.backbone.relu(x)
         x = self.backbone.maxpool(x)
  
         x = self.backbone.layer1(x)
         x = self.backbone.layer2(x)
         x = self.backbone.layer3(x)
         x = self.backbone.layer4(x)
         
         x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
         x = self.out(x)
         
         loss=None
         
         if targets is not None:
             loss = self.loss_fn(x, targets)
             metrics = self.metrics_fn(x, targets)
             return x, loss, metrics
         return x, None, None
     


if __name__=='__main__':
    df = pd.read_csv("D:\\Dataset\\DOGvsCAT\\train_folds.csv")
    df_train=df.loc[df.kfold!=0].reset_index(drop=True)
    df_valid=df.loc[df.kfold==0].reset_index(drop=True)
    
    train_images = df_train.images.values.tolist()
    train_images = [i for i in train_images]
    train_targets = df_train.targets.values
    valid_images = df_valid.images.values.tolist()
    valid_images = [i for i in valid_images]
    valid_targets = df_valid.targets.values
    
    train_aug = A.Compose(
            [
            A.CenterCrop(224,224),
            A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),  
            ]
            
        )
    
    valid_aug = A.Compose(
            [
                A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            ]
        )
    
    
    train_dataset = DogDataset.ClassificationDataset(
            image_paths=train_images,
            targets=train_targets,
            resize=(256,256),
            augmentations=train_aug
        )
    
    valid_dataset = DogDataset.ClassificationDataset(
            image_paths=valid_images,
            targets=valid_targets,
            resize=(256,256),
            augmentations=train_aug
        )
    
    NUM_CLASS=2
    es = EarlyStopping(monitor='valid_loss', model_path=f'model.bin', patience=5, mode="min", delta=0.001)
    modl = CatDogModel(NUM_CLASS, pretrained=True)
    modl.fit(train_dataset, valid_dataset, train_bs=16, valid_bs=16, epochs=10, callback=[es], fp16=True, device='cuda')