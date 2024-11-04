import torch
import os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.dataset_augmentations import createDataset
from transformers import CLIPModel
import yaml
import pandas as pd
from networks import create_architecture, load_weights
import torch.multiprocessing as mp

activations = {}
torch.manual_seed(42)

def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

def runnig_tests(input_csv, device, in_folder, weights_dir, batch_size = 1, N=10):
    table = pd.DataFrame(columns=['filename','clip'])
    dataset = createDataset(in_folder,train=False)
    train_dataset, test_dataset = random_split(dataset, [N, len(dataset) - N])
    
    print("Training images: ",len(train_dataset)," - Test images: ",len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    _, model_path, arch, norm_type, patch_size = get_config("clipdet_latent10k_plus", weights_dir)
    model = load_weights(create_architecture(arch), model_path)
    model = model.to(device).eval()

    print(flush=True)

    ### training
    print("Running the Test")

    with torch.no_grad():
        for batch_idx, (real_images, synthetic_images, real_image_path, tot_synthetic_image_path) in enumerate(tqdm(test_loader)):
            total_images = torch.cat((real_images, synthetic_images), dim=1)
            total_images = torch.squeeze(total_images)
    
            out_tens = model(total_images.clone().to(device)).cpu().numpy()
            
            for ii, logit in zip(np.append(real_image_path, tot_synthetic_image_path), out_tens.flatten()):
                entry = pd.DataFrame.from_dict({
                    "filename": [ii],
                    "clip":  [logit]
                })
                table = pd.concat([table, entry], ignore_index=True)

    print(table)
    return table


if __name__ == "__main__":
    mp.set_start_method('spawn')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug_folder"     , '-a', type=str, help="The path of the dataset folder with the folders of the images' origin", default="../../Baseline/Semantic-Modification-Detection/data/synthbuster")
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images", default="../../Baseline/Semantic-Modification-Detection/data/commercial_tools.csv")
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="./out.csv")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="../../Baseline/Semantic-Modification-Detection/weights")
    parser.add_argument("--models"     , '-m', type=str, help="List of models to test", default='clipdet_latent10k_plus')
    parser.add_argument("--fusion"     , '-f', type=str, help="Fusion function", default='soft_or_prob')
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    args = vars(parser.parse_args())
    
    if args['models'] is None:
        args['models'] = os.listdir(args['weights_dir'])
    else:
        args['models'] = args['models'].split(',')
    
    table = runnig_tests(args['in_csv'], args['device'], args['aug_folder'], args['weights_dir'])
    
    output_csv = args['out_csv']
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    table.to_csv(output_csv, index=False)  # save the results as csv file

    print("Test completed")