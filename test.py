import torch
import os
import pandas
import numpy as np
import tqdm
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import random
import yaml
from networks import create_architecture, load_weights
from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize
from utils.fusion import apply_fusion
from transformers import CLIPModel
from networks.shallow import TwoRegressor, ThreeRegressor
from copy import deepcopy
import re
from torch.utils.tensorboard import SummaryWriter

activations = {}
seed = 42
hidden_size = 512

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']

def test(input_csv, device, N, model_type, batch_size = 1):
    rootdataset = os.path.dirname(os.path.abspath(input_csv))

    table = pandas.read_csv(input_csv)
    train_real_sample_indices = table.iloc[-1000:].sample(n=N, random_state=seed).index
    test_real_sample_indices = table.iloc[-1000:].index.difference(train_real_sample_indices)
    real_sample = table.loc[test_real_sample_indices].reset_index(drop=True)

    global location
    model = None
    regresser = None
    if(model_type[0] == '1'):
        print("Linear SVM")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
        regresser = nn.Linear(1024, 1).to(device)
    elif(model_type[0] == '2'):
        print("Two layers shallow network")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
        regresser = TwoRegressor(hidden_size).to(device)
        regresser.load_state_dict(torch.load("weights/shallow/"+model_type+".pt"))
    elif(model_type[0] == '3'):
        print("Three layers shallow network")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
        regresser = ThreeRegressor(hidden_size).to(device)
        regresser.load_state_dict(torch.load("weights/shallow/"+model_type+".pt"))
    elif(model_type == "original"):
        print("Original paper network")
        _, model_path, arch, norm_type, patch_size = get_config("clipdet_latent10k", weights_dir="./weights")
        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()
        location = "original"
    else:
        print("Wrong model selection")
        exit()

    transform = list()
    print('input model:', model_type, flush=True)
    transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
    transform.append(CenterCrop((224, 224)))
    
    transform.append(make_normalize("clip"))
    transform = Compose(transform)
    print(flush=True)

    output = pandas.DataFrame(columns=['filename','clip'])

    ### training
    print("Running the Testing")
    batch_img = []
    batch_classes = []
    with torch.no_grad():
        for index in tqdm.tqdm(range(len(real_sample))):
            real_filename = os.path.join(rootdataset, real_sample.loc[index, 'filename'])
            synthetic_filename1 = os.path.join(rootdataset, "synthbuster/dalle2/"+real_sample.loc[index, 'filename'].split('/')[-1])
            synthetic_filename2 = os.path.join(rootdataset, "synthbuster/dalle3/"+real_sample.loc[index, 'filename'].split('/')[-1])
            synthetic_filename3 = os.path.join(rootdataset, "synthbuster/firefly/"+real_sample.loc[index, 'filename'].split('/')[-1])
            synthetic_filename4 = os.path.join(rootdataset, "synthbuster/midjourney-v5/"+real_sample.loc[index, 'filename'].split('/')[-1])
            batch_img.append(transform(Image.open(real_filename).convert('RGB')))
            batch_classes.append(0)
            batch_img.append(transform(Image.open(synthetic_filename1).convert('RGB')))
            batch_img.append(transform(Image.open(synthetic_filename2).convert('RGB')))
            batch_img.append(transform(Image.open(synthetic_filename3).convert('RGB')))
            batch_img.append(transform(Image.open(synthetic_filename4).convert('RGB')))
            batch_classes.append(1)
            batch_classes.append(1)
            batch_classes.append(1)
            batch_classes.append(1)

            batch_img = torch.stack(batch_img, 0)

            labels = torch.tensor(batch_classes, dtype=torch.float32).to(device)
            labels = 2 * labels - 1  # Convert labels from 0/1 to -1/+1

            outputs = None
            if(model_type == "original"):
                outputs = model(batch_img.clone().to(device)).cpu().squeeze().tolist()
            else:
                _ = model.get_image_features(pixel_values=batch_img.clone().to(device)).cpu()
                next_to_last_layer_features = activations['next_to_last_layer'].cpu()
                outputs = regresser(next_to_last_layer_features.to(device)).squeeze().cpu().numpy()

            for ii, logit in zip([real_sample.loc[index, 'filename'],"synthbuster/dalle2/"+real_sample.loc[index, 'filename'].split('/')[-1],"synthbuster/dalle3/"+real_sample.loc[index, 'filename'].split('/')[-1],"synthbuster/firefly/"+real_sample.loc[index, 'filename'].split('/')[-1],"synthbuster/midjourney-v5/"+real_sample.loc[index, 'filename'].split('/')[-1]], outputs):
                entry = pandas.DataFrame.from_dict({
                    "filename": [ii],
                    "clip":  [logit]
                })
                output = pandas.concat([output, entry], ignore_index=True)

            batch_img = []
            batch_classes = []

    # save
    return output


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images", default="data/commercial_tools.csv")
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="./out.csv")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="./weights")
    parser.add_argument("--models"     , '-m', type=str, help="List of models to test", default='clipdet_latent10k_plus')
    parser.add_argument("--N"          , '-n', type=int, help="Size of the training N+N vectors", default=100)
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--model_type"     , '-t', type=str, help="Version of the model to be tested", default='original')
    args = vars(parser.parse_args())
    
    table = test(args['in_csv'], args['device'], args['N'], args['model_type'])

    os.makedirs(os.path.dirname(os.path.abspath("predictions/"+args['model_type']+".csv")), exist_ok=True)
    table.to_csv("predictions/"+args['model_type']+".csv", index=False)  # save the results as csv file
    
    print("Testing completed")