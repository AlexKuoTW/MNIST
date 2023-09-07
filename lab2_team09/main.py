import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import numpy as np
from torchvision import transforms, datasets
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            torch.nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 512),
            torch.nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


def calc_acc(output, target):
    predicted = torch.max(output, 1)[1]
    num_samples = target.size(0)
    num_correct = (predicted == target).sum().item()
    return num_correct / num_samples


def training(model, device, train_loader, criterion, optimizer, train_pic):
    # ===============================
    # TODO 3: switch the model to training mode
    model.train()
    # ===============================
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        # =============================================
        # TODO 4: initialize optimizer to zero gradient
        optimizer.zero_grad()
        # =============================================
        output = model(data)
        # =================================================
        # TODO 5: loss -> backpropagation -> update weights
        loss = criterion(output,target)

        loss.backward()
        optimizer.step()
        # =================================================
        epoch_loss.append(loss.item())
        epoch_acc.append(calc_acc(output, target))

        train_acc += calc_acc(output, target)
        train_loss += loss.item()


    train_acc /= len(train_loader)
    train_loss /= len(train_loader)

    epoch_acc = np.mean(epoch_acc)
    epoch_loss = np.mean(epoch_loss)
    train_pic['loss'].append(epoch_loss)
    train_pic["accuracy"].append(epoch_acc)
    end_time = time.time()
    total_time = end_time - start_time


    return train_acc, train_loss, epoch_acc, epoch_loss


def validation(model, device, valid_loader, criterion, valid_pic):
    epoch_loss= []
    epoch_acc= []
    # ===============================
    # TODO 6: switch the model to validation mode
    model.eval()
    # ===============================
    valid_acc = 0.0
    valid_loss = 0.0

    # =========================================
    # TODO 7: turn off the gradient calculation
    with torch.no_grad():

    # =========================================
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)


            # ================================
            # TODO 8: calculate accuracy, loss
            loss = criterion(output,target)
            epoch_loss.append(loss.item())
            epoch_acc.append(calc_acc(output, target))


            valid_acc += calc_acc(output, target)
            valid_loss += loss.item()
            # ================================


    valid_acc /= len(valid_loader)
    valid_loss /= len(valid_loader)

    epoch_acc = np.mean(epoch_acc)
    epoch_loss = np.mean(epoch_loss)
    valid_pic['loss'].append(epoch_loss)
    valid_pic["accuracy"].append(epoch_acc)

    return valid_acc, valid_loss, epoch_acc, epoch_loss

# Add Source def
class Source_image(Dataset):

    def __init__(self,data,tfm,files = None):
        super(Source_image).__init__()
        self.data = data
        self.transform = tfm
        if files != None:
            self.data = files
        
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self,idx):
        # fname = self.data[idx]
        # im = Image.open(fname)
        # im = self.transform(im)
        # #im = self.data[idx]
        # if self.label is not None:
        #     return im, self.label[idx]
        # else:
        #     return im

        fname = self.data[idx]
        im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = 1 # test has no label
        return im,label

def train_valid_split(data_path, valid_ratio, seed):
    data_set = []
    for root, dirs, file in os.walk(data_path):
        for f in file:
            if f.endswith(".png"):
                data_set.append(os.path.join(root,f))
    valid_set_size = int(valid_ratio * len(data_set)) #0.2*2496
    print(valid_set_size)
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    for i in range(0,valid_set_size):
        dirs = str((valid_set[i])[39:40])
        des_path = f".\\NYCU_DL\Deep_learning\HW2\data\\valid\\{dirs}"
        if not os.path.exists(des_path):
            os.mkdir(des_path)
        shutil.move(valid_set[i],des_path)
    return train_set, valid_set



def main():
    # ==================
    # TODO 9: set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # ==================

    # ========================
    # TODO 10: hyperparamete rs
    # you can add your parameters here
    LEARNING_RATE = 0.001
    BATCH_SIZE = 8
    EPOCHS = 60
    # TRAIN_DATA_PATH = ""
    DATA_PATH = ".\\NYCU_DL\Deep_learning\HW2\data"
    MODEL_PATH = 'model.pt'
    ratio = 0.2
    seed = 0
    train_pic = {"loss" : [], "accuracy" : [], "time" : []}
    val_pic = {"loss" : [], "accuracy" : [], "time" : []}
    best_acc = 0.0
    # ========================


    # ===================
    # TODO 11: transforms
    train_transform = transforms.Compose([
        # may be adding some data augmentations?
        transforms.Resize((128,128)),
        # transforms.CenterCrop(100),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # ===================
    valid_transform = transforms.Compose([
        transforms.Resize((128,128)),
        # transforms.CenterCrop(100),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # =================
    # TODO 12: set up datasets
    # hint: ImageFolder?
    if not os.path.exists(DATA_PATH+f'\\valid'):
        os.mkdir(DATA_PATH+f'\\valid')
        print('Start split the data')
        train_source,valid_source = train_valid_split(os.path.join(DATA_PATH,"train"), ratio, seed)
    train_data=dset.ImageFolder(os.path.join(DATA_PATH,"train"),transform=train_transform)
    valid_data=dset.ImageFolder(os.path.join(DATA_PATH,"valid"),transform=valid_transform)

    # train_data = Source_image(train_source, tfm=train_transform)
    # valid_data = Source_image(valid_source, tfm=valid_transform)
    # =================

    # ============================
    # TODO 13 : set up dataloaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
    # ============================

    # build model, criterion and optimizer
    model = Net().to(device).train()
    # ================================
    # TODO 14: criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # ================================


    # training and validation
    train_acc = [0.0] * EPOCHS
    train_loss = [0.0] * EPOCHS
    valid_acc = [0.0] * EPOCHS
    valid_loss = [0.0] * EPOCHS

    print('Start training...')
    for epoch in range(EPOCHS):
        print(f'epoch {epoch} start...')

        train_acc[epoch], train_loss[epoch], epoch_acc, epoch_loss = training(model, device, train_loader, criterion, optimizer, train_pic)
        valid_acc[epoch], valid_loss[epoch], epoch_acc, epoch_loss = validation(model, device, valid_loader, criterion, val_pic)

        if float(valid_acc[epoch]) > best_acc:
            print ("The best is:", valid_acc[epoch])
            torch.save(model.state_dict(), MODEL_PATH)
            best_acc = valid_acc[epoch]

        print(f'epoch={epoch} train_acc={train_acc[epoch]} train_loss={train_loss[epoch]} valid_acc={valid_acc[epoch]} valid_loss={valid_loss[epoch]}')
    print('Training finished')

    # ==================================
    # TODO 15: save the model parameters
    # best_acc = 0.0
    # if {valid_acc[epoch]} > best_acc:
    #     print ("The best is:", valid_acc[epoch])
    #     torch.save(model.state_dict(), MODEL_PATH)
    # ==================================


    # ========================================
    # TODO 16: draw accuracy and loss pictures
    # lab2_teamXX_accuracy.png, lab2_teamXX_loss.png
    # hint: plt.plo
    x = EPOCHS+1
    print(x)
    print(train_pic["loss"])
    print(val_pic["loss"])
    plt.title("Loss")
    plt.plot(np.arange(1, x, 1), train_pic["loss"], color = 'blue')  #21 = ephoe + 1 
    plt.plot(np.arange(1, x, 1), val_pic["loss"], color = 'yellow')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    #Accuracy
    plt.title("Accuracy")
    plt.plot(np.arange(1, x, 1), train_pic["accuracy"], color = 'blue')
    plt.plot(np.arange(1, x, 1), val_pic["accuracy"], color = 'yellow')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
    # =========================================


if __name__ == '__main__':
    main()