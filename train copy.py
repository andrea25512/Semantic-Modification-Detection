import torch
import os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.dataset_augmentations import createDatasetTest
from transformers import CLIPModel
import pandas

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

def test(device, N, batch_size, input_folder):
    print(input_folder)
    dataset = createDatasetTest(input_folder)

    table = pandas.DataFrame(columns=['filename', 'clip'])

    print("Total images: ",len(dataset))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model = model.to(device).eval()
    model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))

    regresser = nn.Linear(1024, 1).to(device).eval()

    print(flush=True)

    ### training
    print("Running the Training")

    with torch.no_grad():
        for batch_idx, (real_images, synthetic_images, real_names, synth_names) in enumerate(tqdm(loader)):
            total_images = torch.cat((real_images, synthetic_images), dim=0)
            total_images = total_images.view(-1, 3, 224, 224)
    
            _ = model.get_image_features(pixel_values=total_images.clone().to(device)).cpu()
            
            next_to_last_layer_features = activations['next_to_last_layer'].cpu()

            real_images_labels = torch.zeros(real_images.size(0)*real_images.size(1))
            synthetic_images_labels = torch.ones(synthetic_images.size(0)*synthetic_images.size(1))
            
            labels = torch.cat((real_images_labels, synthetic_images_labels), dim=0).to(device)
            labels = 2 * labels - 1  # Convert labels from 0/1 to -1/+1

            outputs = regresser(next_to_last_layer_features.to(device)).squeeze()

            pandas.concat({'filename':})

    # save
    torch.save(regresser.state_dict(), 'weights/linear_SVM/SVM.pt')


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder"     , '-i', type=str, help="The path of the dataset folder with the folders of the images' origin", default="../ClipBased-SyntheticImageDetection/data/synthbuster")
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="./results.csv")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="./weights")
    parser.add_argument("--models"     , '-m', type=str, help="List of models to test", default='clipdet_latent10k_plus,Corvi2023')
    parser.add_argument("--N"          , '-n', type=int, help="Size of the training N+N vectors", default=10)
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--batch"          , '-b', type=int, help="Size of the batch duing training and evaluation", default=1)
    args = vars(parser.parse_args())
    
    if args['models'] is None:
        args['models'] = os.listdir(args['weights_dir'])
    else:
        args['models'] = args['models'].split(',')
    
    train(args['device'], args['N'], args["batch"], args['in_folder'])
    
    print("Training completed")