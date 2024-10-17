from torchvision import datasets
from torchvision import transforms
from . import processing
import os
from PIL import Image
import random
import cv2
import numpy as np
from torchvision.transforms import v2
from torchsr.models import ninasr_b1
from torchvision import transforms as T
import torchvision.transforms.functional as F
import torch
import io

class MedianBlurTransform:
    def __init__(self, ksize=3):
        self.ksize = ksize 

    def __call__(self, img):
        img = np.array(img)
        img = cv2.medianBlur(img, self.ksize)
        return Image.fromarray(img)

class BoxBlurTransform:
    def __init__(self, ksize=(3, 3)):
        self.ksize = ksize 

    def __call__(self, img):
        img = np.array(img)
        img = cv2.blur(img, self.ksize)
        return Image.fromarray(img)

class SuperResolutionTransform:
    def __init__(self, upscale_factor=3, device='cuda'):
        self.model = ninasr_b1(pretrained=True, scale=upscale_factor).to(device)
        self.model.eval()  
        self.device = device
        self.upscale_factor = upscale_factor

    def __call__(self, img):
        img_tensor = T.ToTensor()(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            sr_img_tensor = self.model(img_tensor)

        sr_img = T.ToPILImage()(sr_img_tensor.squeeze(0).cpu())
        
        return sr_img

class JPEGCompressionTransform:
    def __init__(self, quality=100):
        self.quality = quality

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        
        with io.BytesIO() as buffer:
            img.save(buffer, format="JPEG", quality=self.quality)
            compressed_img = Image.open(buffer).convert('RGB')

        return compressed_img

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

class GrayscaleToRgb:
    def __call__(self, img):
        return img.convert('RGB')

class WhiteDarkCorrectionTransform:
    def __init__(self, low_percentile=1, high_percentile=99):
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile

    def __call__(self, img):
        img_tensor = F.to_tensor(img) if isinstance(img, Image.Image) else img

        low_val = torch.quantile(img_tensor, self.low_percentile / 100.0)
        high_val = torch.quantile(img_tensor, self.high_percentile / 100.0)

        img_tensor = torch.clamp(img_tensor, min=low_val, max=high_val)
        img_tensor = (img_tensor - low_val) / (high_val - low_val) 

        return F.to_pil_image(img_tensor) 

class LightShadowAdjustmentTransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img) if isinstance(img, Image.Image) else img.numpy().transpose(1, 2, 0)

        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l_channel_clahe = clahe.apply(l_channel)

        lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        return Image.fromarray(img_clahe)

augmentations = [
    v2.RandomEqualize(p=1.0),
    v2.GaussianBlur(kernel_size=9),
    MedianBlurTransform(ksize=9),
    BoxBlurTransform(ksize=(9, 9)),
    v2.Resize(size=(144,144)),
    SuperResolutionTransform(upscale_factor=3),
    transforms.Compose([v2.Grayscale(), GrayscaleToRgb()]),
    JPEGCompressionTransform(quality=5),
    v2.RandomAdjustSharpness(sharpness_factor=5,p=1.0),
    GammaCorrectionTransform(gamma=1.0,gain=1.0),
    WhiteDarkCorrectionTransform(low_percentile=10, high_percentile=90),
    LightShadowAdjustmentTransform(clip_limit=2.0, tile_grid_size=(8, 8))
]

captions = [
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

class ImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform = None, augmentations=None):
        super().__init__(root, transform = transform)
        self.augmentations = augmentations
        self.imgs = self.samples  
        self.real_folder = os.path.join(root, 'real_RAISE_1k')
        self.synthetic_folders = [
            os.path.join(root, 'dalle2'),
            os.path.join(root, 'dalle3'),
            os.path.join(root, 'midjourney-v5'),
            os.path.join(root, 'firefly')
        ]

        self.real_images = [ f for f in os.listdir(self.real_folder) if f.endswith('.png') ]

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, index):
        real_image_name = self.real_images[index]
        real_image_path = os.path.join(self.real_folder, real_image_name)

        real_image = Image.open(real_image_path).convert('RGB')
        real_images = [real_image]

        if self.augmentations:
            for aug in self.augmentations:
                real_images.append(aug(real_image))
        if self.transform:
            for i in range(len(real_images)):
                real_images[i] = self.transform(real_images[i])

        synthetic_folder = random.choice(self.synthetic_folders)
        synthetic_image_path = os.path.join(synthetic_folder, real_image_name)

        synthetic_image = Image.open(synthetic_image_path).convert('RGB')
        synthetic_images = [synthetic_image]

        if self.augmentations:
            for aug in self.augmentations:
                synthetic_images.append(aug(synthetic_image))
        if self.transform:
            for i in range(len(synthetic_images)):
                synthetic_images[i] = self.transform(synthetic_images[i])

        return torch.stack(real_images), torch.stack(synthetic_images)

def createDataset(rootdataset):
    transform = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.CenterCrop((224, 224)),
                                        processing.make_normalize("clip")])
    dataset = ImageDataset(root=rootdataset, transform=transform, augmentations=augmentations)

    return dataset

#-----------------------

class ImageDatasetTest(datasets.ImageFolder):
    def __init__(self, root, transform = None, augmentations=None):
        super().__init__(root, transform = transform)
        self.augmentations = augmentations
        self.imgs = self.samples  
        self.real_folder = os.path.join(root, 'real_RAISE_1k')
        self.synthetic_folders = [
            os.path.join(root, 'dalle2'),
            os.path.join(root, 'dalle3'),
            os.path.join(root, 'midjourney-v5'),
            os.path.join(root, 'firefly')
        ]

        self.real_images = [ f for f in os.listdir(self.real_folder) if f.endswith('.png') ]

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, index):
        real_image_name = self.real_images[index]
        real_image_path = os.path.join(self.real_folder, real_image_name)

        real_image = Image.open(real_image_path).convert('RGB')
        real_images = [real_image]

        if self.augmentations:
            for aug in self.augmentations:
                real_images.append(aug(real_image))
        if self.transform:
            for i in range(len(real_images)):
                real_images[i] = self.transform(real_images[i])

        tot_synthetic_images = torch.Tensor([])
        tot_synthetic_image_path = []
        for synthetic_folder in self.synthetic_folders:
            synthetic_image_path = os.path.join(synthetic_folder, real_image_name)

            synthetic_image = Image.open(synthetic_image_path).convert('RGB')
            synthetic_images = [synthetic_image]

            if self.augmentations:
                for aug in self.augmentations:
                    synthetic_images.append(aug(synthetic_image))
            if self.transform:
                for i in range(len(synthetic_images)):
                    synthetic_images[i] = self.transform(synthetic_images[i])
                    tot_synthetic_image_path.append(synthetic_image_path)

            tot_synthetic_images = torch.cat(tot_synthetic_images,torch.stack(synthetic_images))

        return torch.stack(real_images), tot_synthetic_images, real_image_path, torch.Tensor(tot_synthetic_image_path)

def createDatasetTest(rootdataset):
    transform = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.CenterCrop((224, 224)),
                                        processing.make_normalize("clip")])
    dataset = ImageDatasetTest(root=rootdataset, transform=transform, augmentations=augmentations)

    return dataset

#-----------------------

def show_images_with_captions(images, captions, nrow=4):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(nrow, (len(images) + nrow - 1) // nrow, figsize=(10, 10))
    fig.tight_layout()
    axes = axes.flatten()  

    for i, (img, caption) in enumerate(zip(images, captions)):
        
        img = img.permute(1, 2, 0).numpy()  
        axes[i].imshow(img)
        axes[i].set_title(caption)
        axes[i].axis('off') 

    plt.show()