import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import glob
import cv2
import os
import numpy as np
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torchvision
import matplotlib.pyplot as plt


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv11 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv12 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding="same")

        self.conv21 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv22 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding="same")

        self.conv31 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv32 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="same")

        self.conv41 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv42 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding="same")

        self.conv51 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv52 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding="same")

        self.pool = nn.MaxPool2d(2, 2)

        self.upconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        self.upconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        self.upconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        self.upconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(2, 2), stride=(2, 2))

        self.conv61 = nn.Conv2d(in_channels=256+128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv62 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding="same")

        self.conv71 = nn.Conv2d(in_channels=128+64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv72 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="same")

        self.conv81 = nn.Conv2d(in_channels=64+32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv82 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding="same")

        self.conv91 = nn.Conv2d(in_channels=32+16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.conv92 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding="same")

        self.convout = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1), stride=(1, 1))


    def forward(self, x):

        x = torch.relu(self.conv11(x))
        x1 = torch.relu(self.conv12(x))
        x = self.pool(x1)

        x = torch.relu(self.conv21(x))
        x2 = torch.relu(self.conv22(x))
        x = self.pool(x2)

        x = torch.relu(self.conv31(x))
        x3 = torch.relu(self.conv32(x))
        x = self.pool(x3)

        x = torch.relu(self.conv41(x))
        x4 = torch.relu(self.conv42(x))
        x = self.pool(x4)

        x = torch.relu(self.conv51(x))
        x = torch.relu(self.conv52(x))

        x = self.upconv1(x)
        x = torch.cat((x, x4), dim=1)

        x = torch.relu(self.conv61(x))
        x = torch.relu(self.conv62(x))

        x = self.upconv2(x)
        x = torch.cat((x, x3), dim=1)

        x = torch.relu(self.conv71(x))
        x = torch.relu(self.conv72(x))

        x = self.upconv3(x)
        x = torch.cat((x, x2), dim=1)

        x = torch.relu(self.conv81(x))
        x = torch.relu(self.conv82(x))

        x = self.upconv4(x)
        x = torch.cat((x, x1), dim=1)

        x = torch.relu(self.conv91(x))
        x = torch.relu(self.conv92(x))

        x = torch.sigmoid(self.convout(x))

        return x


class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.data = []
        for img_file in glob.glob(f"{img_dir}\\*.png"):
            self.data.append([img_file, os.path.join(mask_dir, img_file.split("\\")[-1])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128)) / 255.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (128, 128)) / 255.0
        mask = np.expand_dims(mask, axis=0)
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.permute(img_tensor, (2, 0, 1))
        mask_tensor = torch.from_numpy(mask)

        return img_tensor.float(), mask_tensor.float()


def train_val_dataset(dataset, val_split):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, shuffle=False)
    datasets = {}

    train_idx = [index for i, index in enumerate(train_idx)]
    datasets["train"] = Subset(dataset, train_idx)
    datasets["val"] = Subset(dataset, val_idx)
    return datasets



def plot_img(test_loader):

    figure = plt.figure(figsize=(10, 8))

    cols = 3
    rows = 1

    dataiter = iter(test_loader)
    x_test, y_test = dataiter.next()
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    output_test = model(x_test)

    figure.add_subplot(rows, cols, 1)
    x_test = x_test.cpu().numpy() * 255
    x_test = np.squeeze(x_test)
    plt.imshow(np.transpose(x_test, (1, 2, 0)).astype(np.uint8), cmap="gray")

    figure.add_subplot(rows, cols, 2)
    y_test = y_test.cpu().numpy() * 255
    y_test = np.squeeze(y_test)
    plt.imshow(y_test.astype(np.uint8), cmap="gray")

    figure.add_subplot(rows, cols, 3)
    output_test = output_test.detach().cpu().numpy() * 255
    output_test = np.squeeze(output_test)
    plt.imshow(output_test.astype(np.uint8), cmap="gray")

    plt.show()


if __name__ == "__main__":

    device = "cuda:0"

    img_dir = "..\\train\\images"
    ann_dir = "..\\train\\masks"

    n_epochs = 1
    batch_size = 16
    learning_rate = 0.001
    momentum = 0.99

    dataset = CustomDataset(img_dir, ann_dir)
    dataset = train_val_dataset(dataset, val_split=0.10)

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    test_loader = DataLoader(dataset["val"], batch_size=1, shuffle=False, drop_last=True, num_workers=4)

    n_samples_train = len(dataset["train"])

    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = Unet().to(device)
    optim = SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(n_epochs):
        running_loss = 0.0
        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = F.binary_cross_entropy(output, y)

            running_loss += loss
            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"epoch {epoch}, loss: {running_loss/n_samples_train:.4f}")

    plot_img(test_loader)
    torch.save(model.state_dict(), "unet.pth")











