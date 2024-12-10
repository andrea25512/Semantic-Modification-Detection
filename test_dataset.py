from transformers import CLIPModel
from PIL import Image
from torchvision import transforms
import kornia.geometry.transform as TR
import kornia.augmentation as K
import kornia.enhance as E
import torch
from torch import nn

activations = {}

class SpatialRegressor(nn.Module):
    def __init__(self, embedding_dim=1024, num_patches=16):
        super(SpatialRegressor, self).__init__()

        grid_dim = int(num_patches**0.5) 
        self.conv1 = nn.Conv2d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)  
        self.upsample = nn.Upsample(scale_factor=grid_dim, mode='bilinear', align_corners=False)

    def forward(self, x):
        # reshape to [batch_size, embedding_dim, grid_height, grid_width]
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  
        # image of shape [batch_size, 1, grid_height, grid_width]
        print(x.shape)
        x = self.upsample(x)  
        # return [batch_size, img_height, img_width]
        return x.squeeze(1)  

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

transform = torch.nn.Sequential(TR.Resize((224, 224), interpolation='bicubic'), K.CenterCrop((224, 224)), E.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),std=(0.26862954, 0.26130258, 0.27577711)))

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model = model.to("cuda:1").eval()
model.vision_model.encoder.layers[23].register_forward_hook(get_activation('layer23_features'))

image = Image.open("/home/andrea.ceron/Baseline/Semantic-Modification-Detection/data/synthbuster/real_RAISE_1k/r0a2e85f0t.png").convert("RGB")
image = transforms.ToTensor()(image).to("cuda:1").unsqueeze(0)
image = transform(image)

print(image.shape)

_ = model.get_image_features(pixel_values=image)

layer16_features = activations["layer23_features"][0].float()
layer16_features = layer16_features[:, :256, :]
print(layer16_features.shape)

spatial_features = layer16_features.view(layer16_features.shape[0], 16, 16, 1024)

print(spatial_features.shape)

spatial_regressor = SpatialRegressor(embedding_dim=1024, num_patches=196)
spatial_regressor.to("cuda:1")

heatmaps = spatial_regressor(spatial_features)
print(heatmaps.shape)