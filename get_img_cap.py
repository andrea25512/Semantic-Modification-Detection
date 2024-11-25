import torch
import os
import pandas
import tqdm
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import random
from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize
from transformers import CLIPModel
from networks.shallow import TwoRegressor, ThreeRegressor, FourRegressor, FiveRegressor
from transformers import AutoModel
from transformers import CLIPImageProcessor
from torch.utils.tensorboard import SummaryWriter
import subprocess
import pty
from LongCLIP.model import longclip
import csv

activations = {}
seed = 42
hidde_size = 512

def train(input_csv, device, N, layers, weights_dir, learning_rate, version, batch_size = 1):
    torch.manual_seed(seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rootdataset = os.path.join(script_dir,input_csv)
    data_folder = os.path.dirname(os.path.abspath(rootdataset))
    table = pandas.read_csv(rootdataset)
    real_sample_indices = table.iloc[-1000:].sample(n=N, random_state=seed).index
    random.seed(seed)
    real_sample = table.loc[real_sample_indices].reset_index(drop=True)

    file = pandas.read_csv("/home/andrea.ceron/Baseline/Semantic-Modification-Detection/data/synthbuster/prompts.csv")

    print(file)

    ### training
    print("Running the Training")
    #best_model = None
    out = []
    for index in tqdm.tqdm(range(len(real_sample))):
        name = real_sample.loc[index, 'filename'].split('/')[-1][:-4]
        prompt = file[file['image name (matching Raise-1k)'] == name]["Prompt"].item()
        out.append((name, prompt))
    print(out[0])

    df = pandas.DataFrame(out, columns=['ID', 'Prompt'])
    df.to_csv('output.csv', index=False)


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images", default="data/commercial_tools.csv")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="weights/shallow/")
    parser.add_argument("--N"          , '-n', type=int, help="Size of the training N+N vectors", default=100)
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--layers"     , '-l', type=int, help="Number of layers of the regressor", default=4)
    parser.add_argument("--learning_rate"     , '-r', type=float, help="Learning rate of the optimizer", default=0.0005)
    parser.add_argument("--version"     , '-v', type=str, help="Version of the feature extractor", default='LLM2CLIP')
    args = vars(parser.parse_args())
    
    train(args['in_csv'], args['device'], args['N'], args['layers'], args['weights_dir'], args['learning_rate'], args['version'])