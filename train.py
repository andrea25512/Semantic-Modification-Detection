import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.dataset_augmentations import createDataset
from transformers import CLIPModel
from networks.shallow import TwoRegressor, ThreeRegressor, FourRegressor, FiveRegressor
from torch.utils.tensorboard import SummaryWriter
import os
from transformers import AutoModel
from LongCLIP.model import longclip
import kornia.geometry.transform as TR
import kornia.augmentation as K
import kornia.enhance as E
from transformers import CLIPImageProcessor
import subprocess
import pty

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

    if(version == "classic"):
        print('input resize:', '224x224', flush=True)
        transform = torch.nn.Sequential(TR.Resize((224, 224), interpolation='bicubic'), K.CenterCrop((224, 224)), E.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),std=(0.26862954, 0.26130258, 0.27577711)))
    elif(version == "LLM2CLIP"):
        print('input resize:', '336x336', flush=True)
        transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    elif(version == "long"):
        print('input resize:', '224x224', flush=True)
    print(flush=True)

    dataset = createDataset(rootdataset, transform, device, True, False, training_mode, N, seed)
    
    print("Training images: ",len(dataset))

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

    print(flush=True)

    ### training
    print("Running the Training")
    for idx, (real_images, synthetic_images, tot_real_entries, tot_fake_entries) in enumerate(tqdm(train_loader)):
        total_images = torch.cat((real_images, synthetic_images), dim=0)
        total_images = total_images.view(-1, 3, 224, 224)

        if(version == "long"):
            _ = model.encode_image(total_images.clone().to(device))
        else:
            _ = model.get_image_features(pixel_values=total_images.clone().to(device))
        
        next_to_last_layer_features = activations['next_to_last_layer']

        real_images_labels = torch.zeros(real_images.size(0)*real_images.size(1))
        synthetic_images_labels = torch.ones(synthetic_images.size(0)*synthetic_images.size(1))
        
        labels = torch.cat((real_images_labels, synthetic_images_labels), dim=0).to(device)
        labels = 2 * labels - 1  # Convert labels from 0/1 to -1/+1

        outputs = regresser(next_to_last_layer_features.float()).squeeze()

        loss = torch.mean(torch.clamp(1 - outputs * labels, min=0))
        writer.add_scalar('training loss', loss, idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # save
    writer.close()
    torch.save(regresser.state_dict(), os.path.join(script_dir, weights_dir)+name+'.pt')

    import gc
    del model
    del transform
    del regresser
    del dataset
    del train_loader

    torch.cuda.empty_cache()
    gc.collect()

    return name


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images", default="../../Baseline/Semantic-Modification-Detection/data/commercial_tools.csv")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="weights/shallow/")
    parser.add_argument("--N"          , '-n', type=int, help="Size of the training N+N vectors", default=100)
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:1')
    parser.add_argument("--layers"     , '-l', type=int, help="Number of layers of the regressor", default=3)
    parser.add_argument("--learning_rate"     , '-r', type=float, help="Learning rate of the optimizer", default=0.005)
    parser.add_argument("--version"     , '-v', type=str, help="Version of the feature extractor", default='classic')
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