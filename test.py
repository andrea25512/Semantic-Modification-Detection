import torch
import os
import pandas
from tqdm import tqdm
import torch.nn as nn
import yaml
from networks import create_architecture, load_weights
from torch.utils.data import DataLoader, random_split
from transformers import CLIPModel
from networks.shallow import TwoRegressor, ThreeRegressor, FourRegressor, FiveRegressor
from utils.dataset_augmentations import createDataset
from transformers import AutoModel
from transformers import CLIPImageProcessor
from LongCLIP.model import longclip
import random
import kornia.geometry.transform as TR
import kornia.augmentation as K
import kornia.enhance as E
import subprocess
import sys

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

def test(input_csv, device, N, model_type, is_forlab, is_augmentated):
    torch.manual_seed(seed)
    random.seed(seed)
    # making the script runnable from anywhere
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rootdataset = os.path.join(script_dir,input_csv)

    if("LLM2CLIP" not in model_type and "LongClip" not in model_type):
        # here the standard CLIP model is loaded into VRAM, and the next to last feature extractor is initialized
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
    elif(model_type == "original"):
        # here the CLIP+SVM model from the Verdoliva paper is loaded into VRAM (it automatically outputs the next to last features)
        _, model_path, arch, norm_type, patch_size = get_config("clipdet_latent10k", weights_dir=os.path.join(script_dir,"weights"))
        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()
    elif("LLM2CLIP" in model_type):
        # here the LLM2CLIP model is loaded into VRAM, and the next to last feature extractor is initialized
        model = AutoModel.from_pretrained("microsoft/LLM2CLIP-Openai-L-14-336", torch_dtype=torch.float16, trust_remote_code=True)
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
    elif("LongClip" in model_type):
        # here the LongCLIP model is loaded into VRAM, and the next to last feature extractor is initialized
        model, transform = longclip.load(os.path.join(script_dir, "LongCLIP/checkpoints/longclip-L.pt"), device=device)
        model = model.to(device).eval()
        model.visual.ln_post.register_forward_hook(get_activation('next_to_last_layer'))
    else:
        raise ValueError("Could not identify type of feature extractor")  
    
    # one of the possible five network configurations is declared and then loaded also into VRAM
    if("1_layers" in model_type):
        regresser = nn.Linear(1024, 1).to(device)
    elif("2_layers" in model_type):
        regresser = TwoRegressor(1024, hidden_size).to(device)
    elif("3_layers" in model_type):
        regresser = ThreeRegressor(1024, hidden_size).to(device)
    elif("4_layers" in model_type):
        regresser = FourRegressor(1024, hidden_size).to(device)
    elif("5_layers" in model_type):
        regresser = FiveRegressor(1024, hidden_size).to(device)
    else:
        raise ValueError("Could not identify depth of shallow model")
    regresser.load_state_dict(torch.load(os.path.join(script_dir, "weights/shallow/"+model_type+".pt"), weights_only=True))

    # here the image preprocessing is selected based on the CLIP feature extractor version. In the case of LongCLIP the preprocesser is derived when the model is instantiated (thus that's why it cannot be found here below)
    if("LLM2CLIP" in model_type):
        transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    elif("LongClip" in model_type):
        pass
    else:
        transform = torch.nn.Sequential(TR.Resize((224, 224), interpolation='bicubic'), K.CenterCrop((224, 224)), E.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),std=(0.26862954, 0.26130258, 0.27577711)))

    # here a custom dataset is instatiated, indicathing wether to apply or not agumentations, other than passing other parameters like the seed
    test_dataset = createDataset(rootdataset=rootdataset, transform=transform, device=device, train=False, debug=False, training_mode=is_forlab, N=N, seed=seed, do_augs=is_augmentated)
    # batch of dimension one for VRAM limits
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # here the CSV from the predited logits is declared
    output = pandas.DataFrame(columns=['filename','clip','aug'])

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("Running the Testing")
    with torch.no_grad():
        for index, (real_images, synthetic_images, tot_real_entries, tot_fake_entries) in enumerate(tqdm(test_loader, dynamic_ncols=True, file=sys.stderr)):
            total_images = torch.cat((real_images.squeeze(), synthetic_images.squeeze()), dim=0)

            if(model_type == "original"):
                # call to the model in the Verdoliva paper 
                outputs = model(total_images.clone().to(device)).cpu().squeeze().tolist()
            elif("LongClip" in model_type):
                # call to the LongCLIP model 
                _ = model.encode_image(total_images.clone().to(device))
                next_to_last_layer_features = activations['next_to_last_layer']
                outputs = regresser(next_to_last_layer_features.float()).squeeze().cpu().numpy()
            else:
                # call to the CLIP and LLM2CLIP models
                _ = model.get_image_features(pixel_values=total_images.clone().to(device))
                next_to_last_layer_features = activations['next_to_last_layer']
                outputs = regresser(next_to_last_layer_features.float()).squeeze().cpu().numpy()

            # here the logits are paired with the image name and augmentation, in order to do in another file the metric calculation. The table will be saved to a CSV file
            # even if we use the FORLAB dataset instead of the real_RAISE_1k, we still save the real images under the real_RAISE_1k name in order to have back compatibility with the metrics computation
            for idx, (ii, logit) in enumerate(zip(tot_real_entries + tot_fake_entries, outputs)):
                aug_idx = idx % len(test_dataset.captions)
                # select the correct augmentation name
                aug_name = test_dataset.captions[aug_idx]  
                
                entry = pandas.DataFrame.from_dict({
                    "filename": [ii][0],
                    "clip": [logit],
                    "aug": [aug_name]
                })
                if output.empty:
                    output = entry
                else:
                    output = pandas.concat([output, entry], ignore_index=True)

    # return the table
    return output


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images", default="../../Baseline/Semantic-Modification-Detection/data/commercial_tools.csv")
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="1_layers_dataset_SD_0.05_optim_AdamW_N100.csv")
    parser.add_argument("--N"          , '-n', type=int, help="Size of the training N+N vectors", default=990)
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:1')
    parser.add_argument("--model_type"     , '-m', type=str, help="Version of the model to be tested", default='1_layers_dataset_SD_0.05_optim_AdamW_N100')
    parser.add_argument("--forlab"     , '-f', action="store_true", help="Enable testing with FORLAB as the synthetic images (default: False)")
    parser.add_argument("--augmentations"     , '-a', action="store_false", help="Apply augmentations to images (default: True)")
    args = vars(parser.parse_args())
    
    table = test(args['in_csv'], args['device'], args['N'], args['model_type'], args['forlab'], args['augmentations'])

    table.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions/"+args['out_csv']), index=False)  # save the results as csv file
    
    print("Testing completed")
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # here we continue the pipeline by calling automatically the metrics computation on the CSV

    # we reuse most of the parameters that this code recieved from launch, but most importantly we pass the newely constructed CSV name
    params = ["--in_csv", os.path.join(os.path.dirname(os.path.abspath(__file__)), args['in_csv']), "--out_csv", f"predictions/{args['out_csv']}", "--metrics", "auc", "--save_tab", f"performances/{args['out_csv']}"]

    command = ["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)),"compute_metrics.py")] + params
    
    # launch the script and stream its output
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as process:
            for line in process.stdout:
                # print each line in real time
                print(line, end="")
            for err in process.stderr:
                # print errors in real time (if any)
                print(err, end="") 
    except subprocess.CalledProcessError as e:
        print("An error occurred:")
        print(e.stderr)

    #-----------------------------------------------------------------------------------------------------------------------------------------------------
    # here we continue the pipeline by calling automatically an additional metrics computation on the CSV. This one wil lcreate the AUC scores for each specific augmentation

    # we reuse most of the parameters that this code recieved from launch, but most importantly we pass the newely constructed CSV name
    params = ["--in_csv", os.path.join(os.path.dirname(os.path.abspath(__file__)), args['in_csv']), "--out_csv", f"predictions/{args['out_csv']}", "--metrics", "auc", "--save_tab", f"performances/TABLE_{args['out_csv']}"]

    command = ["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)),"augmentation_wise_metrics.py")] + params
    
    # launch the script and stream its output
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as process:
            for line in process.stdout:
                # print each line in real time
                print(line, end="") 
            for err in process.stderr:
                # print errors in real time (if any)
                print(err, end="")  
    except subprocess.CalledProcessError as e:
        print("An error occurred:")
        print(e.stderr)