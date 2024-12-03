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

def test(input_csv, device, N, model_type, test_mode, batch_size = 1):
    torch.manual_seed(seed)
    random.seed(seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rootdataset = os.path.join(script_dir,input_csv)

    if(test_mode == "FORLAB"):
        FORLAB_real_samples = [f for f in os.listdir("/media/mmlab/Volume2/TrueFake/PreSocial/Real/FORLAB") if os.path.isfile(os.path.join("/media/mmlab/Volume2/TrueFake/PreSocial/Real/FORLAB", f))]
        random.shuffle(FORLAB_real_samples)

    model = None
    regresser = None
    if(model_type[0] == '1'):
        print("Linear SVM")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
        regresser = nn.Linear(1024, 1).to(device)
        regresser.load_state_dict(torch.load(os.path.join(script_dir, "weights/shallow/"+model_type+".pt")))
    elif(model_type[0] == '2'):
        print("Two layers shallow network")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
        regresser = TwoRegressor(1024, hidden_size).to(device)
        regresser.load_state_dict(torch.load(os.path.join(script_dir, "weights/shallow/"+model_type+".pt")))
    elif(model_type[0] == '3'):
        print("Three layers shallow network")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
        regresser = ThreeRegressor(1024, hidden_size).to(device)
        regresser.load_state_dict(torch.load(os.path.join(script_dir, "weights/shallow/"+model_type+".pt")))
    elif(model_type[0] == '4'):
        print("Four layers shallow network")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
        regresser = FourRegressor(1024, hidden_size).to(device)
        regresser.load_state_dict(torch.load(os.path.join(script_dir, "weights/shallow/"+model_type+".pt")))
    elif(model_type[0] == '5'):
        print("Five layers shallow network")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
        regresser = FiveRegressor(1024, hidden_size).to(device)
        regresser.load_state_dict(torch.load(os.path.join(script_dir, "weights/shallow/"+model_type+".pt")))
    elif(model_type == "original"):
        print("LLM2CLIP network")
        _, model_path, arch, norm_type, patch_size = get_config("clipdet_latent10k", weights_dir=os.path.join(script_dir,"weights"))
        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()
    elif(model_type[0] == "L" and model_type[1] == "L"):
        print("Microsoft model")
        model = AutoModel.from_pretrained("microsoft/LLM2CLIP-Openai-L-14-336", torch_dtype=torch.float16, trust_remote_code=True)
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
        if(model_type[9] == "1"):
            print("Linear SVM")
            regresser = nn.Linear(1024, 1).to(device)
        elif(model_type[9] == "2"):
            print("Two layers shallow network")
            regresser = TwoRegressor(1024, hidden_size).to(device)
        elif(model_type[9] == "3"):
            print("Three layers shallow network")
            regresser = ThreeRegressor(1024, hidden_size).to(device)
        elif(model_type[9] == "4"):
            print("Four layers shallow network")
            regresser = FourRegressor(1024, hidden_size).to(device)
        elif(model_type[9] == "5"):
            print("Five layers shallow network")
            regresser = FiveRegressor(1024, hidden_size).to(device)
        else:
            print("Could not identify depth of shallow model")
            exit()
        regresser.load_state_dict(torch.load(os.path.join(script_dir, "weights/shallow/"+model_type+".pt")))
    elif(model_type[0] == "L" and model_type[1] == "o"):
        print("LongCLIP model")
        model, transform = longclip.load(os.path.join(script_dir, "LongCLIP/checkpoints/longclip-L.pt"), device=device)
        model = model.to(device).eval()
        model.visual.ln_post.register_forward_hook(get_activation('next_to_last_layer'))
        if(model_type[9] == "1"):
            print("Linear SVM")
            regresser = nn.Linear(1024, 1).to(device)
        elif(model_type[9] == "2"):
            print("Two layers shallow network")
            regresser = TwoRegressor(1024, hidden_size).to(device)
        elif(model_type[9] == "3"):
            print("Three layers shallow network")
            regresser = ThreeRegressor(1024, hidden_size).to(device)
        elif(model_type[9] == "4"):
            print("Four layers shallow network")
            regresser = FourRegressor(1024, hidden_size).to(device)
        elif(model_type[9] == "5"):
            print("Five layers shallow network")
            regresser = FiveRegressor(1024, hidden_size).to(device)
        else:
            print("Could not identify depth of shallow model")
            exit()
        regresser.load_state_dict(torch.load(os.path.join(script_dir, "weights/shallow/"+model_type+".pt")))
    else:
        print("Wrong model selection")
        exit()    

    if(model_type[0] == "L" and model_type[1] == "L"):
        print('input resize:', '336x336', flush=True)
        transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    elif(model_type[0] == "L" and model_type[1] == "o"):
        print('input resize:', '224x224', flush=True)
    else:
        print('input resize:', '224x224', flush=True)
        transform = torch.nn.Sequential(TR.Resize((224, 224), interpolation='bicubic'), K.CenterCrop((224, 224)), E.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),std=(0.26862954, 0.26130258, 0.27577711)))
    print(flush=True)

    if("dataset_SD" in model_type):
        training_mode = "SD"
    else:
        training_mode = "original"

    test_dataset = createDataset(rootdataset, transform, device, False, False, training_mode, N, seed)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    print("Testing images: ", len(test_dataset))

    output = pandas.DataFrame(columns=['filename','clip','aug'])

    ### training
    print("Running the Testing")
    with torch.no_grad():
        for index, (real_images, synthetic_images, tot_real_entries, tot_fake_entries) in enumerate(tqdm(test_loader, dynamic_ncols=True, file=sys.stderr)):
            total_images = torch.cat((real_images.squeeze(), synthetic_images.squeeze()), dim=0)

            labels = torch.tensor(torch.cat((torch.zeros(real_images.shape[1]), torch.ones(synthetic_images.shape[1])), dim=0), dtype=torch.float32).to(device)
            labels = 2 * labels - 1  # Convert labels from 0/1 to -1/+1

            outputs = None
            if(model_type == "original"):
                outputs = model(total_images.clone().to(device)).cpu().squeeze().tolist()
            elif(model_type[0] == "L" and model_type[1] == "o"):
                _ = model.encode_image(total_images.clone().to(device))
                next_to_last_layer_features = activations['next_to_last_layer']
                outputs = regresser(next_to_last_layer_features.float()).squeeze().cpu().numpy()
            else:
                _ = model.get_image_features(pixel_values=total_images.clone().to(device))
                next_to_last_layer_features = activations['next_to_last_layer']
                outputs = regresser(next_to_last_layer_features.float()).squeeze().cpu().numpy()

            for idx, (ii, logit) in enumerate(zip(tot_real_entries + tot_fake_entries, outputs)):
                aug_idx = idx % len(test_dataset.captions)
                aug_name = test_dataset.captions[aug_idx]  # Select the correct augmentation name
                
                entry = pandas.DataFrame.from_dict({
                    "filename": [ii][0],
                    "clip": [logit],
                    "aug": [aug_name]
                })
                output = pandas.concat([output, entry], ignore_index=True)

    # save
    return output


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images", default="../../Baseline/Semantic-Modification-Detection/data/commercial_tools.csv")
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="1_layers_dataset_SD_0.05_optim_AdamW_N100.csv")
    parser.add_argument("--N"          , '-n', type=int, help="Size of the training N+N vectors", default=990)
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:1')
    parser.add_argument("--model_type"     , '-m', type=str, help="Version of the model to be tested", default='1_layers_dataset_SD_0.05_optim_AdamW_N100')
    parser.add_argument("--test_mode"     , '-t', type=str, help="RAISE1k or FORLAB as the real images to test with", default='FORLAB')
    args = vars(parser.parse_args())
    
    table = test(args['in_csv'], args['device'], args['N'], args['model_type'], args['test_mode'])

    table.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions/"+args['out_csv']), index=False)  # save the results as csv file
    
    print("Testing completed")

    # Define the parameters to pass
    params = ["--in_csv", os.path.join(os.path.dirname(os.path.abspath(__file__)), args['in_csv']), "--out_csv", f"predictions/{args['out_csv']}", "--metrics", "auc", "--save_tab", f"performances/{args['out_csv']}"]

    # Construct the command
    command = ["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)),"compute_metrics.py")] + params
    print(command)
    # Launch the script and stream its output
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="")  # Print each line in real time
            for err in process.stderr:
                print(err, end="")  # Print errors in real time (if any)
    except subprocess.CalledProcessError as e:
        print("An error occurred:")
        print(e.stderr)

    #-----------------------------------------------------------------------------------------------------------------------------------------------------

    # Define the parameters to pass
    params = ["--in_csv", os.path.join(os.path.dirname(os.path.abspath(__file__)), args['in_csv']), "--out_csv", f"predictions/{args['out_csv']}", "--metrics", "auc", "--save_tab", f"performances/TABLE_{args['out_csv']}"]

    # Construct the command
    command = ["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)),"augmentation_wise_metrics.py")] + params
    print(command)
    # Launch the script and stream its output
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="")  # Print each line in real time
            for err in process.stderr:
                print(err, end="")  # Print errors in real time (if any)
    except subprocess.CalledProcessError as e:
        print("An error occurred:")
        print(e.stderr)