import torch
import os
import pandas
import numpy as np
import tqdm
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import random

from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize
from utils.fusion import apply_fusion
from transformers import CLIPModel
from networks.shallow import TwoRegressor, ThreeRegressor
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

activations = {}
seed = 42
hidde_size = 512

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

def train(input_csv, device, N, layers, weights_dir, learning_rate, batch_size = 1):
    rootdataset = os.path.dirname(os.path.abspath(input_csv))

    table = pandas.read_csv(input_csv)
    real_sample_indices = table.iloc[-1000:].sample(n=N, random_state=seed).index
    random.seed(seed)
    real_sample = table.loc[real_sample_indices].reset_index(drop=True)

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model = model.to(device).eval()
    model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))

    regresser = None
    if(layers == 1):
        regresser = nn.Linear(1024, 1).to(device)
    elif(layers == 2):
        regresser = TwoRegressor(hidde_size).to(device)
    elif(layers == 3):
        regresser = ThreeRegressor(hidde_size).to(device)
    else:
        print("Only supported from 1 to 3 layers")
        exit()
    

    optimizer = optim.AdamW(regresser.parameters(), lr=learning_rate)

    transform = list()
    print('input resize:', 'Clip224', flush=True)
    transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
    transform.append(CenterCrop((224, 224)))
    
    transform.append(make_normalize("clip"))
    transform = Compose(transform)
    print(flush=True)

    name = None
    if(layers == 1):
        name = '1_layers_'+str(learning_rate)+'_optim_'+str(type(optimizer).__name__)+'_N'+str(N)
    else:
        name = str(layers)+'_layers_'+str(hidde_size)+'_ReLU_'+str(learning_rate)+'_optim_'+str(type(optimizer).__name__)+'_N'+str(N)
    print("Output name: ",name)
    writer = SummaryWriter('runs/'+name)

    ### training
    print("Running the Training")
    batch_img = []
    batch_classes = []
    best_loss = float('inf')
    #best_model = None
    for index in tqdm.tqdm(range(len(real_sample))):
        real_filename = os.path.join(rootdataset, real_sample.loc[index, 'filename'])
        synthetic_choice = random.choice(["dalle2", "dalle3", "firefly", "midjourney-v5"])
        synthetic_filename = os.path.join(rootdataset, "synthbuster/"+synthetic_choice+"/"+real_sample.loc[index, 'filename'].split('/')[-1])
        batch_img.append(transform(Image.open(real_filename).convert('RGB')))
        batch_classes.append(0)
        batch_img.append(transform(Image.open(synthetic_filename).convert('RGB')))
        batch_classes.append(1)

        batch_img = torch.stack(batch_img, 0)

        _ = model.get_image_features(pixel_values=batch_img.clone().to(device)).cpu()
        
        next_to_last_layer_features = activations['next_to_last_layer'].cpu()

        labels = torch.tensor(batch_classes, dtype=torch.float32).to(device)
        labels = 2 * labels - 1  # Convert labels from 0/1 to -1/+1

        outputs = regresser(next_to_last_layer_features.to(device)).squeeze()

        loss = torch.mean(torch.clamp(1 - outputs * labels, min=0))
        writer.add_scalar('training loss', loss, index)
        if(best_loss > loss):
            best_loss = loss
            #best_model = deepcopy(regresser.state_dict())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_img = []
        batch_classes = []

    # save
    writer.close()
    print("best loss: ",best_loss)
    torch.save(regresser.state_dict(), weights_dir+name+'.pt')


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images", default="data/commercial_tools.csv")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="weights/shallow/")
    parser.add_argument("--N"          , '-n', type=int, help="Size of the training N+N vectors", default=100)
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--layers"     , '-l', type=int, help="Number of layers of the regressor", default=3)
    parser.add_argument("--learning_rate"     , '-r', type=float, help="Learning rate of the optimizer", default=0.001)
    args = vars(parser.parse_args())
    
    train(args['in_csv'], args['device'], args['N'], args['layers'], args['weights_dir'], args['learning_rate'])
    
    print("Training completed")