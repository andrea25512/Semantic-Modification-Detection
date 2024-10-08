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

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

def train(input_csv, device, N, batch_size = 10):
    rootdataset = os.path.dirname(os.path.abspath(input_csv))

    table = pandas.read_csv(input_csv)
    real_sample_indices = table.iloc[-1000:].sample(n=N, random_state=42).index
    synthetic_sample_indices = []
    for real_index in real_sample_indices:
        relative_position = real_index - 4000
        block_choice = random.choice([0, 1000, 2000, 3000])
        synthetic_index = block_choice + relative_position
        synthetic_sample_indices.append(synthetic_index)

    real_sample = table.loc[real_sample_indices]
    synthetic_sample = table.loc[synthetic_sample_indices]

    combined_train = pandas.concat([real_sample, synthetic_sample])
    combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model = model.to(device).eval()
    model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))

    classifier = nn.Linear(1024, 1).to(device)

    learning_rate = 0.01
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)

    transform = list()
    print('input resize:', 'Clip224', flush=True)
    transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
    transform.append(CenterCrop((224, 224)))
    
    transform.append(make_normalize("clip"))
    transform = Compose(transform)
    print(flush=True)

    ### training
    print("Running the Training")
    batch_img = []
    batch_classes = []
    batch_id = list()
    last_index = combined_train.index[-1]
    for index in tqdm.tqdm(combined_train.index, total=len(combined_train)):
        filename = os.path.join(rootdataset, combined_train.loc[index, 'filename'])
        type = combined_train.loc[index, 'typ']
        batch_img.append(transform(Image.open(filename).convert('RGB')))
        batch_classes.append(0 if type=='real' else 1)
        batch_id.append(index)

        if (len(batch_id) >= batch_size) or (index==last_index):
            batch_img = torch.stack(batch_img, 0)

            _ = model.get_image_features(pixel_values=batch_img.clone().to(device)).cpu()
            
            next_to_last_layer_features = activations['next_to_last_layer'].cpu()

            labels = torch.tensor(batch_classes, dtype=torch.float32).to(device)
            labels = 2 * labels - 1  # Convert labels from 0/1 to -1/+1

            outputs = classifier(next_to_last_layer_features.to(device)).squeeze()

            loss = torch.mean(torch.clamp(1 - outputs * labels, min=0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_img = []
            batch_classes = []
            batch_id = list()

    # save
    torch.save(classifier.state_dict(), 'weights/linear_SVM/SVM.pt')


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images")
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="./results.csv")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="./weights")
    parser.add_argument("--models"     , '-m', type=str, help="List of models to test", default='clipdet_latent10k_plus,Corvi2023')
    parser.add_argument("--N"          , '-n', type=int, help="Size of the training N+N vectors", default=10)
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    args = vars(parser.parse_args())
    
    if args['models'] is None:
        args['models'] = os.listdir(args['weights_dir'])
    else:
        args['models'] = args['models'].split(',')
    
    train(args['in_csv'], args['device'], args['N'])
    
    print("Training completed")