# %%
# %%capture
# !pip install --quiet imutils
# !pip install --quiet wget
# !pip install --quiet split-folders
# !pip install --quiet timm
# !pip install --quiet pytorch-lightning
# !pip install --quiet torchmetrics
# !pip install --quiet rich

# %%
from imutils import paths
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import splitfolders
from torch import nn
import numpy as np
import os
import torchmetrics
import timm
import wget
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import cv2
import tensorflow as tf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# %% [markdown]
# ## Downloading Dataset

# %%
_URL = 'http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz'
wget.download(_URL)


# %%
zip_dir = tf.keras.utils.get_file('./logo', origin=_URL, untar=True,extract=True)

# %%
import tarfile

fname = 'flickr_logos_27_dataset.tar.gz'

if fname.endswith("tar.gz"):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()

# %%
fname = 'flickr_logos_27_dataset/flickr_logos_27_dataset_images.tar.gz'

if fname.endswith("tar.gz"):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()

# %%
src_dir = "flickr_logos_27_dataset_images"
dest = "LOGOS"

if not os.path.exists(dest):
    os.makedirs(dest)

# %% [markdown]
# ## Preprocessing

# %%
import pandas as pd

# %%
df = pd.read_csv("flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt", sep='\s+',header=None)

# %%
df

# %%
X = df.iloc[:,0]
Y = df.iloc[:,1]

# %%
dtdir = './flickr_logos_27_dataset_images/'

# %%
im = df[0][0]

# %%
size = df.iloc[:,3:]

# %%
size

# %%
img = os.path.join(dtdir,im)

# %%
size = size.values.tolist()

# %%
size[0][0],size[0][1],size[0][2],size[0][3]

# %%
image = cv2.imread(img)
plt.imshow(image)
image.shape

# %%
image = cv2.imread(img)
image = image[size[0][1]:size[0][3],size[0][0]:size[0][2]]
plt.imshow(image)
image.shape

# %%
query = pd.read_csv("./flickr_logos_27_dataset/flickr_logos_27_dataset_query_set_annotation.txt", sep='\s+',header=None)

# %%
query

# %%
img = os.path.join(dtdir,query[0][5])
image = cv2.imread(img)
plt.imshow(image)
plt.axis("off")
image.shape

# %%
y = list(set(list(Y)))
y.sort()

# %%
for i in y:
    os.makedirs(os.path.join(dest,i))

# %%
distractor = pd.read_csv("./flickr_logos_27_dataset/flickr_logos_27_dataset_distractor_set_urls.txt", sep='\s+',header=None)

# %%
distractor

# %%
HEIGHT = 224
WIDTH =  224

# %% [markdown]
# ## Removing Corrupt Images 

# %%
for i in range(len(X)):
    try:
        destrain = os.path.join(dest,Y[i])
        savepath = os.path.join(destrain,X[i])
        img  = os.path.join(dtdir,X[i])
        image = cv2.imread(img)
        image = image[size[i][1]:size[i][3],size[i][0]:size[i][2]]
        image = cv2.resize(image,(WIDTH,HEIGHT))
        cv2.imwrite(savepath,image)
    except:
        print('error')
        pass

# %%
A = query.iloc[:,0]
B = query.iloc[:,1]

# %%
A

# %%

for i in range(len(A)):
    try:
        destrain = os.path.join(dest,B[i])
        savepath = os.path.join(destrain,A[i])
        img  = os.path.join(dtdir,A[i])
        image = cv2.imread(img)
        image = cv2.resize(image,(WIDTH,HEIGHT))
        cv2.imwrite(savepath,image)
    except:
        print('error')
        pass


# %%
imagePaths = list(paths.list_images(dest))

# %%
img = imagePaths[40]
print(img)
image = cv2.imread(img)
plt.imshow(image)
plt.axis("off")
image.shape

# %% [markdown]
# ## Train Val Split

# %%
path = 'LOGOS'

# %%
splitfolders.ratio(path, output="data", seed=42, ratio=(0.8,0.2))

# %% [markdown]
# ## Image Augmentation

# %%
# initialize our data augmentation functions
resize = transforms.Resize(size=(WIDTH,HEIGHT))
hFlip = transforms.RandomHorizontalFlip(p=0.25)
vFlip = transforms.RandomVerticalFlip(p=0.25)
rotate = transforms.RandomRotation(degrees=15)
coljtr = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)
raf = transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15)
rrsc = transforms.RandomResizedCrop(size=WIDTH, scale=(0.8, 1.0))
ccp  = transforms.CenterCrop(size=WIDTH)  # Image net standards
nrml = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  # Imagenet standards

# %%
# initialize our training and validation set data augmentation
# pipeline
trainTransforms = transforms.Compose([resize,hFlip,vFlip,rotate,raf,rrsc,ccp,coljtr,transforms.ToTensor(),nrml])
valTransforms = transforms.Compose([resize,transforms.ToTensor(),nrml])

# %%
BS = 256
# initialize the training and validation dataset
print("[INFO] loading the training and validation dataset...")
trainDataset = ImageFolder(root='./data/train',transform=trainTransforms)
valDataset = ImageFolder(root='./data/val', transform=valTransforms)
print("[INFO] training dataset contains {} samples...".format(len(trainDataset)))
print("[INFO] validation dataset contains {} samples...".format(len(valDataset)))

# %%
class LitNeuralNet(pl.LightningModule):
    def __init__(self,num_classes):
        super(LitNeuralNet, self).__init__()
        
        
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        self.model.aux_logits=False

        # Freeze training for all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.classifier = nn.Sequential(
                      nn.Linear(self.model.classifier.in_features, 512), 
                      nn.BatchNorm1d(512),
                      nn.Dropout(0.4),
                      nn.ReLU(inplace=True),
                      nn.Linear(512, 256), 
                      nn.BatchNorm1d(256),
                      nn.Dropout(0.4),
                      nn.ReLU(inplace=True),
                      nn.Linear(256, num_classes),
                      nn.Softmax())
        # add metrics
        self.acc = torchmetrics.Accuracy(task="multiclass",num_classes=num_classes)
        
    def forward(self, x):
        out = self.model(x)
        return out

    def train_dataloader(self):
        trainDataLoader = DataLoader(trainDataset, num_workers=2,batch_size=BS, shuffle=True)

        return trainDataLoader

    def val_dataloader(self):
        valDataLoader = DataLoader(valDataset, num_workers=2,batch_size=BS,shuffle=False)
        
        return valDataLoader

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # Forward pass
        outputs = self.forward(images)
        lossfn = nn.CrossEntropyLoss()
        loss = lossfn(outputs, labels)
        
        #y_pred = torch.exp(outputs)
        y_pred = torch.argmax(outputs,dim=1)
        #y_pred = output.data.max(1, keepdim=True)[1]
        train_acc = self.acc(y_pred, labels)
        # just accumulate

        self.log("train_loss", loss)
        self.log("train_accuracy", train_acc)
        tensorboard_logs = {'train_loss': loss}
        # use key 'log'
        return {"loss": loss, 'log': tensorboard_logs}
    
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        # Forward pass
        outputs = self.forward(images)
        lossfn = nn.CrossEntropyLoss()   
        loss = lossfn(outputs, labels)
        
        #pred = torch.exp(outputs)
        y_pred = torch.argmax(outputs,dim=1)
        #pred = output.data.max(1, keepdim=True)[1]
        self.acc.update(y_pred, labels)

        self.log("val_loss", loss)
        return {"val_loss": loss}
            
    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_accuracy = self.acc.compute()
        # log metrics
        self.log("val_accuracy", val_accuracy)
        self.log("val_loss", avg_loss)
        # reset all metrics
        self.acc.reset()
        print(f"\nVal Accuracy: {val_accuracy:.4} "\
        f"Val Loss: {avg_loss:.4}")
        
        tensorboard_logs = {'avg_val_loss': avg_loss}
        # use key 'log'
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())


# %%
# seeding for torch, numpy, stdlib random, including DataLoader workers!
seed_everything(123, workers=True)

early_stopping = EarlyStopping(
    monitor="val_loss",
    stopping_threshold=1e-5,
    divergence_threshold=9.0,
    check_finite=True)

# %%
#from pytorch_lightning.loggers import WandbLogger

# %%
#wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

# %%
model = LitNeuralNet(num_classes=len(trainDataset.classes))
trainer = Trainer(accelerator='gpu', devices=-1,max_epochs=100,log_every_n_steps=8)#,callbacks=[early_stopping])
trainer.fit(model)

# %% [markdown]
# ## Prediction On Test Images

# %%
testimage = list(paths.list_images('./flickr_logos_27_dataset_images'))

# %% [markdown]
# ![image.png](attachment:dbacbc64-b039-4beb-877c-5061780c5dc3.png)

# %%
def predimg(path):
    from PIL import Image
    image = Image.open(path)
    plt.imshow(image)
    plt.axis("off")
    plt.show() 
    model.eval()
    with torch.no_grad():
      img =  load_img(path)
      mean = [0.485, 0.456, 0.406] 
      std = [0.229, 0.224, 0.225]
      transform_norm = transforms.Compose([transforms.ToTensor(), 
      transforms.Resize((224,224)),transforms.Normalize(mean, std)])
      img_normalized = transform_norm(img).float()
      img_normalized = img_normalized.unsqueeze(0)
      img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)
      img = img_normalized#.to(device)
      img = DataLoader(img, num_workers=4,batch_size=BS,shuffle=False)
      output = trainer.predict(model,img)
      output = output[0][0]
      #print(output)
      index = output.data.cpu().numpy().argmax()
      result = list(np.around(output.data.cpu().numpy()*100,1))
      print(result)
      print("PREDICTED CLASS = ",trainDataset.classes[index])

# %%
predimg(testimage[25])

# %%
predimg(testimage[40])



