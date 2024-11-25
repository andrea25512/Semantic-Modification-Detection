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
import kornia.filters as KF
import kornia.color as C

U.get_cuda_device_if_available(index=0)

#------------------------------------------------------------------------------------------------------------

class EqualizeTransform:
    def __init__(self):
        pass

    def __call__(self, img_tensor):
        return E.equalize(img_tensor)

class GaussianBlurTransform:
    def __init__(self, kernel_size_range=(3, 9), sigma_range=(0.1, 2.0), debug=False):
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
        self.debug = debug

    def __call__(self, img_tensor):
        # Ensure kernel size is odd and within the specified range
        kernel_sizes = [k for k in range(self.kernel_size_range[0], self.kernel_size_range[1]+1, 2)]
        kernel_size = random.choice(kernel_sizes)
        sigma = random.uniform(*self.sigma_range)
        if(self.debug):
            print(f"GaussianBlurTransform: selected {kernel_size} kernel size out of {kernel_sizes} kernel sizes - selected {sigma} sigma range out of {self.sigma_range} sigma range")
        return KF.gaussian_blur2d(img_tensor, (kernel_size, kernel_size), (sigma, sigma)).squeeze(0)

class MedianBlurTransform:
    def __init__(self, kernel_size_range=(3, 5), debug=False):
        self.kernel_size_range = kernel_size_range
        self.debug = debug

    def __call__(self, img_tensor):
        kernel_sizes = [k for k in range(self.kernel_size_range[0], self.kernel_size_range[1]+1, 2)]
        kernel_size = random.choice(kernel_sizes)
        if(self.debug):
            print(f"MedianBlurTransform: selected {kernel_size} kernel size out of {kernel_sizes} kernel sizes")
        return KF.median_blur(img_tensor, (kernel_size, kernel_size)).squeeze(0)

class BoxBlurTransform:
    def __init__(self, kernel_size_range=(3, 7), debug=False):
        self.kernel_size_range = kernel_size_range
        self.debug = debug

    def __call__(self, img_tensor):
        kernel_sizes = [k for k in range(self.kernel_size_range[0], self.kernel_size_range[1]+1, 2)]
        kernel_size = random.choice(kernel_sizes)
        if(self.debug):
            print(f"BoxBlurTransform: selected {kernel_size} kernel size out of {kernel_sizes} kernel sizes")
        return KF.box_blur(img_tensor, (kernel_size, kernel_size)).squeeze(0)

class ResizeTransform:
    def __init__(self, size_range=(0.3, 0.7), debug=False):
        self.size_range = size_range
        self.debug = debug

    def __call__(self, img_tensor):
        size = random.uniform(*self.size_range)
        if(self.debug):
            print(f"ResizeTransform: from {img_tensor.shape[2]}x{img_tensor.shape[3]}--({size})-->{int(img_tensor.shape[2]*size)}x{int(img_tensor.shape[3]*size)} size out of {self.size_range} sizes")
        return torch.nn.functional.interpolate(img_tensor, size=(int(img_tensor.shape[2]*size), int(img_tensor.shape[3]*size)), mode='bilinear', align_corners=False).squeeze(0)

class SuperResolutionTransform:
    def __init__(self, upscale_factor=3, device='cuda'):
        self.model = ninasr_b1(pretrained=True, scale=upscale_factor).to(device)
        self.model.eval()
        self.device = device

    def __call__(self, img_tensor):
        with torch.no_grad():
            sr_img_tensor = self.model(img_tensor)
        return sr_img_tensor

class JPEGCompressionTransform:
    def __init__(self, quality_range=(10, 40), debug=False):
        self.quality_range = quality_range
        self.debug = debug

    def __call__(self, img_tensor):
        quality = random.randint(*self.quality_range)
        local_img_tensor = img_tensor.clone().cpu().squeeze(0)
        img_pil = F.to_pil_image(local_img_tensor)
        with io.BytesIO() as buffer:
            img_pil.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            compressed_img = Image.open(buffer).convert('RGB')
        compressed_tensor = F.to_tensor(compressed_img).to(img_tensor.device)
        if(self.debug):
          print(f"JPEGCompressionTransform: selected {quality} quality out of {self.quality_range} qualities")
        return compressed_tensor.unsqueeze(0)

class SharpnessTransform:
    def __init__(self, sharpness_range=(2.0, 5.0), debug=False):
        self.sharpness_range = sharpness_range
        self.debug = debug

    def __call__(self, img_tensor):
        factor = random.uniform(*self.sharpness_range)
        if(self.debug):
            print(f"SharpnessTransform: selected {factor} sharpness out of {self.sharpness_range} sharpness ranges")
        return E.sharpness(img_tensor, factor)

class GammaTransform:
    def __init__(self, gamma_range=(0.7, 1.5), gain_range=(0.7, 1.5), debug=False):
        self.gamma_range = gamma_range
        self.gain_range = gain_range
        self.debug = debug

    def __call__(self, img_tensor):
        gamma = random.uniform(*self.gamma_range)
        gain = random.uniform(*self.gain_range)
        if(self.debug):
            print(f"GammaTransform: selected {gamma} gamma out of {self.gamma_range} gamma ranges - selected {gain} gain out of {self.gain_range} gain ranges")
        return E.adjust_gamma(img_tensor, gamma=gamma, gain=gain)

class WhiteDarkCorrectionTransform:
    def __init__(self, low_percentile_range=(10, 30), high_percentile_range=(70, 90), debug=False):
        self.low_percentile_range = low_percentile_range
        self.high_percentile_range = high_percentile_range
        self.debug = debug

    def __call__(self, img_tensor):
        low_percentile = random.uniform(*self.low_percentile_range)
        high_percentile = random.uniform(*self.high_percentile_range)
        low_val = torch.quantile(img_tensor, low_percentile / 100.0)
        high_val = torch.quantile(img_tensor, high_percentile / 100.0)
        img_tensor = torch.clamp(img_tensor, min=low_val, max=high_val)
        img_tensor = (img_tensor - low_val) / (high_val - low_val + 1e-8)
        if(self.debug):
            print(f"WhiteDarkCorrectionTransform: selected {low_percentile} low percentile out of {self.low_percentile_range} low percentile range - selected {high_percentile} high percentile out of {self.high_percentile_range} high percentile range")
        return img_tensor

class LightShadowAdjustmentTransform:
    def __init__(self, clip_limit_range=(1.0, 4.0), grid_size_options=[(4, 4), (8, 8), (16, 16)], debug=False):
        self.clip_limit_range = clip_limit_range
        self.grid_size_options = grid_size_options
        self.debug = debug

    def __call__(self, img_tensor):
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        clip_limit = random.uniform(*self.clip_limit_range)
        grid_size = random.choice(self.grid_size_options)
        img_clahe = E.equalize_clahe(
            img_tensor.unsqueeze(0),
            clip_limit=clip_limit,
            grid_size=grid_size,
            slow_and_differentiable=False
        ).squeeze(0)
        if(self.debug):
            print(f"WhiteDarkCorrectionTransform: selected {clip_limit} clip limit out of {self.clip_limit_range} clip limit range - selected {grid_size} grid size out of {self.grid_size_options} grid size options")
        return img_clahe

#------------------------------------------------------------------------------------------------------------

class ImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform = None, device="cuda:0", train=False, debug=False):
        super().__init__(root, transform = transform)
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

        self.augmentations = [
            EqualizeTransform(),
            GaussianBlurTransform(debug=debug),
            MedianBlurTransform(debug=debug),
            BoxBlurTransform(debug=debug),
            ResizeTransform(debug=debug),
            SuperResolutionTransform(device=device),
            K.RandomGrayscale(p=1.0),
            JPEGCompressionTransform(debug=debug),
            SharpnessTransform(debug=debug),
            GammaTransform(debug=debug),
            WhiteDarkCorrectionTransform(debug=debug),
            LightShadowAdjustmentTransform(debug=debug)
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
        real_image = transforms.ToTensor()(real_image).to(self.device).unsqueeze(0)
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
            synthetic_image = transforms.ToTensor()(synthetic_image).to(self.device).unsqueeze(0)
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

def createDataset(rootdataset, device="cuda:0", train=False, debug=False):
    transform = torch.nn.Sequential(TR.Resize((224, 224), interpolation='bicubic'),
                                        K.CenterCrop((224, 224)),
                                        E.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),std=(0.26862954, 0.26130258, 0.27577711)))
    dataset = ImageDataset(root=rootdataset, transform=transform, device=device, train=train, debug=debug)

    return dataset

#------------------------------------------------------------------------------------------------------------

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