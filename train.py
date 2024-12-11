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

def train(input_csv, device, N, layers, weights_dir, learning_rate, version, is_stable_diffusion):
    torch.manual_seed(seed)
    random.seed(seed)
    # making the script runnable from anywhere
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rootdataset = os.path.join(script_dir,input_csv)
    data_folder = os.path.dirname(os.path.abspath(rootdataset))
    # since no augmentations are needed, we can read the images directly. Thus here we load the image dictionary
    table = pandas.read_csv(rootdataset)
    # sampled N real images from the dictionary. The semi-random sampling is done on the last 1000 entries since the image dictionary is constructed as [synthetic1, synthetic2, synthetic3, synthetic4, real], where each subset is of size 1000 (5000 entries total)
    real_sample_indices = table.iloc[-1000:].sample(n=N, random_state=seed).index
    # a semi-random shuffle of the selected N indexes is applied
    real_samples = table.loc[real_sample_indices].reset_index(drop=True)

    # here the selection of the feature extraction is applied, where "classic" is the original CLIP model, "LLM2CLIP" is the LLM2CLIP model and finally the "long" is the LongCLIP model
    # in the original Verdoliva paper they utilized the feature just before the final layer (projection to a smaller dimension), here is replicated via a "register_forward_hook"
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
        raise ValueError("Wrong model selection. Supported models [classic, LLM2CLIP, long]")

    # here is selected the size of the shallow network that will input the feature of one of the CLIP versions, and output a single value in [-1, +1]
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
        raise ValueError("Wrong network depth selection. Supported depths [1, 2, 3, 4, 5]")
    
    optimizer = optim.AdamW(regresser.parameters(), lr=learning_rate)

    # here the iamge preprocessing is selected based on the CLIP feature extractor version. In the case of LongCLIP the preprocesser is derived when the model is instantiated (thus that's why it cannot be found here below)
    if(version == "classic"):
        transform = list()
        transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
        transform.append(CenterCrop((224, 224)))
        transform.append(make_normalize("clip"))
        transform = Compose(transform)
    elif(version == "LLM2CLIP"):
        transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    elif(version == "long"):
        pass

    # the naming convetion of the experiments changes if we change a derivation of CLIP
    name = ""
    if(version == "LLM2CLIP"):
        name += "LLM2CLIP_"
    elif(version == "long"):
        name += "LongClip_"

    # also the model name changes if we utilize an alternative dataset
    # this dataset is present in order to make possible to detect a shift in dataset distribution during testing (here we change the synthtic dataset, during testing we change the real dataset)
    tmp = ""
    if(is_stable_diffusion):
        tmp = "dataset_SD_"

    # here the rest of the naming is created loggin multiple variables
    if(layers == 1):
        name += f"1_layers_{tmp}{learning_rate}_optim_{type(optimizer).__name__}_N{N}"
    else:
        name += f"{layers}_layers_{tmp}{hidde_size}_ReLU_{learning_rate}_optim_{type(optimizer).__name__}_N{N}"
    print("Output name: ",name)
    writer = SummaryWriter('runs/'+name)

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("Running the Training")
    # batch size is fized to one since we load directly the images without creating a dataset + dataloader instances
    batch_img = []
    batch_classes = []
    for index in tqdm.tqdm(range(len(real_samples))):
        # the real file name under the real_RAISE_1k is extracted.
        real_filename = os.path.join(data_folder, real_samples.loc[index, 'filename'])
        
        # the synthetic image is loaded under the normal or alternative dataset
        if(is_stable_diffusion):
            synthetic_filename = os.path.join("/media/mmlab/Datasets_4TB/ceron_train/StableDiffusion35/no_PP/"+f"{index:05}.png")
        else:
            synthetic_choice = random.choice(["dalle2", "dalle3", "firefly", "midjourney-v5"])
            synthetic_filename = os.path.join(data_folder, "synthbuster/"+synthetic_choice+"/"+real_samples.loc[index, 'filename'].split('/')[-1])
        
        # the real and synthetic images are put in the same batch, as well to the labels 
        batch_img.append(transform(Image.open(real_filename).convert('RGB')))
        batch_classes.append(-1)
        batch_img.append(transform(Image.open(synthetic_filename).convert('RGB')))
        batch_classes.append(1)

        # here the LLM2CLIP preprocesser gives back the images in another more accessible format, thus the extraction via "item['pixel_values'][0]" is not needed
        if(not version == "LLM2CLIP"):
            batch_img = torch.stack(batch_img, 0)
        else:
            batch_img = [item['pixel_values'][0] for item in batch_img]
            batch_img = torch.stack([torch.tensor(arr) for arr in batch_img])

        # the LongCLIP feature extractor has another nomenclature to actualyl extract the feature in respect of CLIP and LLM2CLIP
        if(version == "long"):
            _ = model.encode_image(batch_img.clone().to(device))
        else:
            _ = model.get_image_features(pixel_values=batch_img.clone().to(device))
        
        # the feature are extracted before the down projection, thus remaining at the bigger dimensionality (1024)
        next_to_last_layer_features = activations['next_to_last_layer']

        # the shallow network recieves the 1024 dimensional features
        outputs = regresser(next_to_last_layer_features.float()).squeeze()
        
        # the loss in calculated via a Hinge loss (choosen arbitrarly since the training code was not shared in the Verdoliva paper)
        labels = torch.tensor(batch_classes, dtype=torch.float32).to(device)
        loss = torch.mean(torch.clamp(1 - outputs * labels, min=0))
        writer.add_scalar('training loss', loss, index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_img = []
        batch_classes = []

    # save
    writer.close()
    torch.save(regresser.state_dict(), os.path.join(script_dir, weights_dir)+name+'.pt')

    return name


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images", default="data/commercial_tools.csv")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="weights/shallow/")
    parser.add_argument("--N"          , '-n', type=int, help="Size of the training N+N vectors (maximum is 1000)", default=100)
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:1')
    parser.add_argument("--layers"     , '-l', type=int, help="Number of layers of the regressor (from 1 to 5)", default=3)
    parser.add_argument("--learning_rate"     , '-r', type=float, help="Learning rate of the optimizer", default=0.005)
    parser.add_argument("--version"     , '-v', type=str, help="Version of the feature extractor (choices: classic, long, LLM2CLIP)", default='classic', choices=["classic", "long", "LLM2CLIP"])
    parser.add_argument("--stable_diffusion"     , '-s', action="store_true", help="Enable training with StableDiffusion as the synthetic images (default: False value)")

    args = vars(parser.parse_args())

    name = train(args['in_csv'], args['device'], args['N'], args['layers'], args['weights_dir'], args['learning_rate'], args['version'], args['stable_diffusion'])
    
    print("Training completed")
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # here we continue the pipeline by calling automatically the evalaution on the remaining dataset

    # we reuse most of the parameters that this code recieved from launch, but most importantly we pass the newely constructed model name
    params = ["--model_type", name, "--N", str(args['N']), "--out_csv", f"{name}.csv", "--device", args['device']]

    # if during training the StableDiffusion datast was utilized (alternative to synthetic), we need to test the distribution shift in another real dataset, in this case the FORLAB was selected
    if(args['stable_diffusion']):
        params = params + ["--forlab"]

    command = ["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)),"test.py")] + params

    # use pty to mimic a terminal
    master_fd, slave_fd = pty.openpty()
    try:
        with subprocess.Popen(command, stdout=slave_fd, stderr=slave_fd, text=True, bufsize=1) as process:
            # close the writing end in this process
            os.close(slave_fd)  
            # read from the pseudo-terminal
            while True:
                try:
                    # read in chunks
                    output = os.read(master_fd, 1024).decode()  
                    if not output:
                        # break if no more output (end of process)
                        break  
                    # print as it comes in
                    print(output, end="")  
                except OSError:
                    # exit the loop gracefully if the pseudo-terminal is closed
                    break  
    finally:
        # close the pseudo-terminal
        try:
            os.close(master_fd)
        except OSError:
            pass
            
    