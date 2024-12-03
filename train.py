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

activations = {}
seed = 42
hidde_size = 512

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

def train(input_csv, device, N, layers, weights_dir, learning_rate, version, training_mode, batch_size = 1):
    torch.manual_seed(seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rootdataset = os.path.join(script_dir,input_csv)
    data_folder = os.path.dirname(os.path.abspath(rootdataset))
    table = pandas.read_csv(rootdataset)
    real_sample_indices = table.iloc[-1000:].sample(n=N, random_state=seed).index
    random.seed(seed)
    real_samples = table.loc[real_sample_indices].reset_index(drop=True)

    if(version == "classic"):
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
    elif(version == "LLM2CLIP"):
        model = AutoModel.from_pretrained("microsoft/LLM2CLIP-Openai-L-14-336", torch_dtype=torch.float16, trust_remote_code=True)
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
    elif(version == "long"):
        model, transform = longclip.load(os.path.join(script_dir, "LongCLIP/checkpoints/longclip-L.pt"), device=device)
        model = model.to(device).eval()
        model.visual.ln_post.register_forward_hook(get_activation('next_to_last_layer'))
    else:
        print("Wrong model selection")
        exit()

    regresser = None
    if(layers == 1):
        regresser = nn.Linear(1024, 1).to(device)
    elif(layers == 2):
        regresser = TwoRegressor(1024, hidde_size).to(device)
    elif(layers == 3):
        regresser = ThreeRegressor(1024, hidde_size).to(device)
    elif(layers == 4):
        regresser = FourRegressor(1024, hidde_size).to(device)
    elif(layers == 5):
        regresser = FiveRegressor(1024, hidde_size).to(device)
    else:
        print("Only supported from 1 to 5 layers")
        exit()
    
    optimizer = optim.AdamW(regresser.parameters(), lr=learning_rate)

    if(version == "classic"):
        transform = list()
        print('input resize:', '224x224', flush=True)
        transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
        transform.append(CenterCrop((224, 224)))
        
        transform.append(make_normalize("clip"))
        transform = Compose(transform)
    elif(version == "LLM2CLIP"):
        print('input resize:', '336x336', flush=True)
        transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    elif(version == "long"):
        print('input resize:', '224x224', flush=True)
    print(flush=True)

    name = ""
    if(version == "LLM2CLIP"):
        name += "LLM2CLIP_"
    elif(version == "long"):
        name += "LongClip_"

    tmp = ""
    if(training_mode == "SD"):
        tmp = "dataset_SD_"

    if(layers == 1):
        name += f"1_layers_{tmp}{learning_rate}_optim_{type(optimizer).__name__}_N{N}"
    else:
        name += f"{layers}_layers_{tmp}{hidde_size}_ReLU_{learning_rate}_optim_{type(optimizer).__name__}_N{N}"
    print("Output name: ",name)
    writer = SummaryWriter('runs/'+name)

    ### training
    print("Running the Training")
    batch_img = []
    batch_classes = []
    best_loss = float('inf')
    #best_model = None
    for index in tqdm.tqdm(range(len(real_samples))):
        real_filename = os.path.join(data_folder, real_samples.loc[index, 'filename'])
        
        if(training_mode == "SD"):
            synthetic_filename = os.path.join("/media/mmlab/Datasets_4TB/ceron_train/StableDiffusion35/no_PP/"+f"{index:05}.png")
        else:
            synthetic_choice = random.choice(["dalle2", "dalle3", "firefly", "midjourney-v5"])
            synthetic_filename = os.path.join(data_folder, "synthbuster/"+synthetic_choice+"/"+real_samples.loc[index, 'filename'].split('/')[-1])
        
        batch_img.append(transform(Image.open(real_filename).convert('RGB')))
        batch_classes.append(0)
        batch_img.append(transform(Image.open(synthetic_filename).convert('RGB')))
        batch_classes.append(1)

        if(not version == "LLM2CLIP"):
            batch_img = torch.stack(batch_img, 0)
        else:
            batch_img = [item['pixel_values'][0] for item in batch_img]
            batch_img = torch.stack([torch.tensor(arr) for arr in batch_img])

        if(version == "long"):
            _ = model.encode_image(batch_img.clone().to(device))
        else:
            _ = model.get_image_features(pixel_values=batch_img.clone().to(device))
        
        next_to_last_layer_features = activations['next_to_last_layer']

        labels = torch.tensor(batch_classes, dtype=torch.float32).to(device)
        labels = 2 * labels - 1  # Convert labels from 0/1 to -1/+1

        outputs = regresser(next_to_last_layer_features.float()).squeeze()

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
    torch.save(regresser.state_dict(), os.path.join(script_dir, weights_dir)+name+'.pt')

    return name


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
    parser.add_argument("--training_mode"     , '-t', type=str, help="RAISE1k with StableDiffusion or the previous version", default="SD")

    args = vars(parser.parse_args())
    
    name = train(args['in_csv'], args['device'], args['N'], args['layers'], args['weights_dir'], args['learning_rate'], args['version'], args['training_mode'])
    
    print("Training completed")
    
    # Define the parameters to pass
    params = ["--model_type", name, "--N", str(args['N']), "--out_csv", f"{name}.csv", "--test_mode", "FORLAB", "--device", args['device']]

    # Construct the command
    command = ["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)),"test.py")] + params

    # Use pty to mimic a terminal
    master_fd, slave_fd = pty.openpty()
    try:
        with subprocess.Popen(command, stdout=slave_fd, stderr=slave_fd, text=True, bufsize=1) as process:
            os.close(slave_fd)  # Close the writing end in this process
            # Read from the pseudo-terminal
            while True:
                try:
                    output = os.read(master_fd, 1024).decode()  # Read in chunks
                    if not output:
                        break  # Break if no more output (end of process)
                    print(output, end="")  # Print as it comes in
                except OSError:
                    break  # Exit the loop gracefully if the pseudo-terminal is closed
    finally:
        # Close the pseudo-terminal
        try:
            os.close(master_fd)
        except OSError:
            pass
            
    