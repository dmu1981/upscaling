"""Upscaling by a factor of 2"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
import torchvision
from tqdm import tqdm

device = "cuda"
BATCH_SIZE = 64

class Upsampler(nn.Module):
    """Upsampler by a factor of 2"""
    def __init__(self):
        super().__init__()

        self.fw = nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2),
          nn.BatchNorm2d(num_features=16),
          nn.ReLU(),
          nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=2),
          nn.BatchNorm2d(num_features=64),
          nn.ReLU(),
          nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, padding=2),
          nn.BatchNorm2d(num_features=256),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=256,out_channels=64,kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(num_features=64),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=64,out_channels=16,kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(num_features=16),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=4, stride=2, padding=1),
          nn.Tanh()
        )

    def forward(self, x):
        return self.fw(x)

class ImageDataset(Dataset):
    def __init__(self, path, image_size):
        self.path = path
        self.imgs = os.listdir(path)

        BORDER = image_size // 8
        self.original = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
              (image_size+BORDER,image_size+BORDER),
              interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.Lambda(lambda x: x.type(torch.float)),
            torchvision.transforms.Normalize(128,134),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomCrop((image_size, image_size)),
        ])

        self.downsampled = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
              (image_size // 2,image_size // 2),
              interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            #AddGaussianNoise(0, 0.05),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = torchvision.io.read_image(os.path.join(self.path, self.imgs[idx]))
        img = self.original(img)

        return self.downsampled(img), img


dataloader = DataLoader(ImageDataset("./train/cats/", 512), batch_size=BATCH_SIZE, shuffle=True)

class Trainer:
    """ The training class for the Upscaler"""
    def __init__(self):
        self.model = Upsampler()
        self.model.to(device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0005)
        self.loss_function = torch.nn.MSELoss()

        self.start_epoch = 0
        try:
            checkpoint = torch.load("upscaler_256to512.pt")

            self.model.\
              load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.\
              load_state_dict(checkpoint['optimizer_state_dict'])

            self.start_epoch = checkpoint['epoch']
        except:
            print("Could not load model from disk, starting from scratch")

    def save_images(self, epoch, inp, upscaled, gt):
        """Generate some images and save them to disk for review"""
        print(inp.size())
        print(upscaled.size())
        print(gt.size())

        inp = torch.nn.Upsample(scale_factor=2, mode='bilinear')(inp[:8,:,:,:])
        img = (torch.cat([inp[:8,:,:,:], upscaled[:8,:,:,:], gt[:8,:,:,:]]) + 1) / 2

        grid = torchvision.utils.make_grid(img, nrow=8)
        im = torchvision.transforms.ToPILImage()(grid)
        im.save("epoch_{}.png".format(epoch))

    def train(self):
        """Train some epochs"""
        it = iter(dataloader)
        for epoch in range(self.start_epoch, 25000):
            bar = tqdm(dataloader)
            total_loss = 0
            cnt = 0
            for batch, gt in bar:
                self.optimizer.zero_grad()
                x = self.model(batch.to(device))
                loss = self.loss_function(x, gt.to(device))
                loss.backward()
                total_loss += loss.item()
                cnt += 1
                self.optimizer.step()
                bar.set_description("epoch {}, loss={:.8f}".format(epoch, 1000.0*total_loss/cnt))

            batch, gt = next(iter(dataloader))
            x = self.model(batch.to(device))
            self.save_images(epoch, batch.to("cpu"), x.to("cpu"), gt.to("cpu"))

            torch.save({
                      'epoch': epoch,
                      'model_state_dict':
                        self.model.state_dict(),
                      'optimizer_state_dict':
                        self.optimizer.state_dict(),
                      }, "upscaler_256to512.pt")

Trainer = Trainer()
Trainer.train()
