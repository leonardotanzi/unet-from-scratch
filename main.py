from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import SGD
from utils import *


if __name__ == "__main__":

    device = "cuda:0"

    img_dir = "..\\train\\images"
    ann_dir = "..\\train\\masks"

    n_epochs = 50
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











