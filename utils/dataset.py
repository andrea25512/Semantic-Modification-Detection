from torchvision import datasets
import os
from PIL import Image, ImageFile
from torchvision import transforms
import numpy as np
import torch

class ImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform = None, device="cuda:0"):
        real_images = root+"BtB_dataset/"
        super().__init__(real_images, transform = transform)
        self.root = real_images
        self.device = device
        
        self.image_files_roots = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    full_path = os.path.join(root, "_".join(file.split('_')[:-1]))
                    self.image_files_roots.append(full_path)

        to_exlude = "/media/mmlab/Volume2/BtB_dataset/FloreView/D41_L4S2C3_medium_0"
        self.image_files_roots.remove(to_exlude)

    def __len__(self):
        return len(self.image_files_roots)

    def __getitem__(self, index):
        image_path = self.image_files_roots[index]
        image_file = f"{image_path}_inpainted-0.png"
        inpainted_image = None
        try:
            inpainted_image = self.transform(Image.open(image_file).convert('RGB'))
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
        

        splitted_path = image_path.split('/')
        splitted_path[-3] = "BtB_masks"
        splitted_path[-1] = f"mask_{'_'.join(splitted_path[-1].split('_')[:-1])}.png"
        splitted_path = '/'.join(splitted_path)
        mask_image = np.array(Image.open(splitted_path).convert('L'))
        ratio_inpainting = np.sum(mask_image == 255) / mask_image.size

        return inpainted_image, ratio_inpainting

def createDataset(rootdataset, device="cuda:0"):
    transform = list()
    transform.append(transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC))
    transform.append(transforms.CenterCrop((224, 224)))
    transform.append(transforms.ToTensor())
    transform.append(
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
        )
    transform = transforms.Compose(transform)
    dataset = ImageDataset(root=rootdataset, transform=transform, device=device)

    return dataset