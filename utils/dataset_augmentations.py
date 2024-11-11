from torchvision import datasets
from torchvision import transforms
from . import processing
import os
from PIL import Image
import random
import cv2
import numpy as np
import torch.nn as nn
from torchvision.transforms import v2
from torchsr.models import ninasr_b1
from torchvision import transforms as T
import torchvision.transforms.functional as F
import torch
import io
import kornia.augmentation as K
import kornia.enhance as E
import kornia.utils as U
import kornia.geometry.transform as TR

U.get_cuda_device_if_available(index=0)

class SuperResolutionTransform:
    def __init__(self, upscale_factor=3, device='cuda'):
        self.model = ninasr_b1(pretrained=True, scale=upscale_factor).to(device)
        self.model.eval()
        self.device = device

    def __call__(self, img_tensor):
        with torch.no_grad():
            sr_img_tensor = self.model(img_tensor)
        return sr_img_tensor


class SuperResolutionTransform:
    def __init__(self, upscale_factor=3, device='cuda'):
        self.model = ninasr_b1(pretrained=True, scale=upscale_factor).to(device)
        self.model.eval()
        self.device = device

    def __call__(self, img_tensor):
        with torch.no_grad():
            sr_img_tensor = self.model(img_tensor.unsqueeze(0))
        return sr_img_tensor.squeeze(0)

class JPEGCompressionTransform:
    def __init__(self, quality=100):
        self.quality = quality

    def __call__(self, img_tensor):
        img_tensor = img_tensor.cpu()
        img_pil = transforms.ToPILImage()(img_tensor)
        with io.BytesIO() as buffer:
            img_pil.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            compressed_img = Image.open(buffer).convert('RGB')
        compressed_tensor = transforms.ToTensor()(compressed_img).to("cuda:0")
        return compressed_tensor


class GammaCorrectionTransform:
    def __init__(self, gamma=1.0, gain=1.0):
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return F.adjust_gamma(img, gamma=self.gamma, gain=self.gain)
        elif isinstance(img, Image.Image):
            img_tensor = F.to_tensor(img)
            img_tensor = F.adjust_gamma(img_tensor, gamma=self.gamma, gain=self.gain)
            return F.to_pil_image(img_tensor)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

class WhiteDarkCorrectionTransform:
    def __init__(self, low_percentile=1, high_percentile=99):
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile

    def __call__(self, img_tensor):
        low_val = torch.quantile(img_tensor, self.low_percentile / 100.0)
        high_val = torch.quantile(img_tensor, self.high_percentile / 100.0)

        img_tensor = torch.clamp(img_tensor, min=low_val, max=high_val)
        img_tensor = (img_tensor - low_val) / (high_val - low_val) 

        return img_tensor

class LightShadowAdjustmentTransform:
    def __init__(self, clip_limit=2.0, grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.grid_size = grid_size

    def __call__(self, img_tensor):
        # Ensure the image is in the range [0, 1]
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

        # Apply CLAHE
        img_clahe = E.equalize_clahe(
            img_tensor, 
            clip_limit=self.clip_limit, 
            grid_size=self.grid_size, 
            slow_and_differentiable=False
        )
        return img_clahe

augmentations = [
    K.RandomEqualize(p=1.0),
    K.RandomGaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0), p=1.0),
    K.RandomMedianBlur((9, 9), p=1.0),
    K.RandomBoxBlur((9, 9), p=1.0),
    K.Resize(size=(144,144), p=1.0),
    SuperResolutionTransform(upscale_factor=3),
    K.RandomGrayscale(p=1.0),
    JPEGCompressionTransform(quality=5),
    K.RandomSharpness(sharpness=1,p=1.0),
    K.RandomGamma(gamma=(1.5, 1.5), gain=(1.5, 1.5), p=1.0),
    WhiteDarkCorrectionTransform(low_percentile=10, high_percentile=90),
    LightShadowAdjustmentTransform(clip_limit=2.0, grid_size=(8, 8))
]

class ImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform = None, augmentations=None, device="cuda:0", train=False):
        super().__init__(root, transform = transform)
        self.augmentations = augmentations
        self.imgs = self.samples  
        self.real_folder = os.path.join(root, 'real_RAISE_1k')
        self.root = root
        self.device = device
        self.train = train
        self.synthetic_folders = [
            'dalle2',
            'dalle3',
            'midjourney-v5',
            'firefly'
        ]
        self.captions = [
            "Original",
            "Equalize",
            "Gaussian",
            "Median",
            "Blur",
            "Resize",
            "S-Resolut",
            "Grayscale",
            "Compres",
            "Shaprn",
            "Gamma",
            "Wht/Drk",
            "Shadow"
        ]

        self.real_images = [ f for f in os.listdir(self.real_folder) if f.endswith('.png') ]

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, index):
        real_image_name = self.real_images[index]
        real_image_path = os.path.join(self.real_folder, real_image_name)

        real_image = Image.open(real_image_path).convert('RGB')
        real_image = transforms.ToTensor()(real_image).to(self.device)
        real_images = [real_image]
        tot_real_entries = []

        if self.augmentations:
            for aug in self.augmentations:
                real_images.append(aug(real_image))

        #show_images_with_captions([img.cpu() for img in real_images],self.captions)

        if self.transform:
            for i in range(len(real_images)):
                real_images[i] = self.transform(real_images[i])
                tot_real_entries.append("synthbuster/real_RAISE_1k/"+real_image_name)

        tot_synthetic_images = torch.Tensor([]).to(self.device)
        tot_fake_entries = []

        tmp = self.synthetic_folders.copy()
        if(self.train):
            tmp = [random.choice(tmp)]

        for synthetic_type in tmp:
            synthetic_folder = os.path.join(self.root, synthetic_type)
            synthetic_image_path = os.path.join(synthetic_folder, real_image_name)

            synthetic_image = Image.open(synthetic_image_path).convert('RGB')
            synthetic_image = transforms.ToTensor()(synthetic_image).to(self.device)
            synthetic_images = [synthetic_image]
                                                                                                            
            if self.augmentations:
                for aug in self.augmentations:
                    synthetic_images.append(aug(synthetic_image))
            if self.transform:
                for i in range(len(synthetic_images)):
                    synthetic_images[i] = self.transform(synthetic_images[i])
                    tot_fake_entries.append("synthbuster/"+synthetic_type+"/"+real_image_name)

            tot_synthetic_images = torch.cat((tot_synthetic_images, torch.stack(synthetic_images)), dim=0)

        return torch.stack(real_images), tot_synthetic_images, tot_real_entries, tot_fake_entries

def createDataset(rootdataset, device="cuda:0", train=False):
    transform = torch.nn.Sequential(TR.Resize((224, 224), interpolation='bicubic'),
                                        K.CenterCrop((224, 224)),
                                        E.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),std=(0.26862954, 0.26130258, 0.27577711)))
    dataset = ImageDataset(root=rootdataset, transform=transform, augmentations=augmentations, device=device, train=train)

    return dataset

#-----------------------

def show_images_with_captions(images, captions, nrow=4):
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(50, 50))
    axes = fig.subplots(nrow, (len(images) + nrow - 1) // nrow)
    fig.tight_layout()
    axes = axes.flatten()  

    for i, (img, caption) in enumerate(zip(images, captions)):
        if(img.dim() == 4):
            img = img.squeeze(0)
        img = img.permute(1, 2, 0).numpy()  
        axes[i].imshow(img)
        axes[i].set_title(caption)
        axes[i].axis('off') 

    fig.savefig("real.png")