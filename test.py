import torch
import os
import pandas
import tqdm
import torch.nn as nn
from PIL import Image
import yaml
from networks import create_architecture, load_weights
from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize
from transformers import CLIPModel
from networks.shallow import TwoRegressor, ThreeRegressor, FourRegressor, FiveRegressor
from transformers import AutoModel
from transformers import CLIPImageProcessor
import sys
from LongCLIP.model import longclip
import subprocess

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
    torch.manual_seed(seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rootdataset = os.path.join(script_dir,input_csv)
    data_folder = os.path.dirname(os.path.abspath(rootdataset))
    table = pandas.read_csv(rootdataset)
    train_real_sample_indices = table.iloc[-1000:].sample(n=N, random_state=seed).index
    test_real_sample_indices = table.iloc[-1000:].index.difference(train_real_sample_indices)
    real_sample = table.loc[test_real_sample_indices].reset_index(drop=True)

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
        transform = list()
        print('input resize:', '224x224', flush=True)
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
        for index in tqdm.tqdm(range(len(real_sample)), dynamic_ncols=True, file=sys.stderr):
            real_filename = os.path.join(data_folder, real_sample.loc[index, 'filename'])
            synthetic_filename1 = os.path.join(data_folder, "synthbuster/dalle2/"+real_sample.loc[index, 'filename'].split('/')[-1])
            synthetic_filename2 = os.path.join(data_folder, "synthbuster/dalle3/"+real_sample.loc[index, 'filename'].split('/')[-1])
            synthetic_filename3 = os.path.join(data_folder, "synthbuster/firefly/"+real_sample.loc[index, 'filename'].split('/')[-1])
            synthetic_filename4 = os.path.join(data_folder, "synthbuster/midjourney-v5/"+real_sample.loc[index, 'filename'].split('/')[-1])
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

            if(not (model_type[0] == "L" and model_type[1] == "L")):
                batch_img = torch.stack(batch_img, 0)
            else:
                batch_img = [item['pixel_values'][0] for item in batch_img]
                batch_img = torch.stack([torch.tensor(arr) for arr in batch_img])

            labels = torch.tensor(batch_classes, dtype=torch.float32).to(device)
            labels = 2 * labels - 1  # Convert labels from 0/1 to -1/+1

            outputs = None
            if(model_type == "original"):
                outputs = model(batch_img.clone().to(device)).cpu().squeeze().tolist()
            elif(model_type[0] == "L" and model_type[1] == "o"):
                _ = model.encode_image(batch_img.clone().to(device))
                next_to_last_layer_features = activations['next_to_last_layer']
                outputs = regresser(next_to_last_layer_features.float()).squeeze().cpu().numpy()
            else:
                _ = model.get_image_features(pixel_values=batch_img.clone().to(device))
                next_to_last_layer_features = activations['next_to_last_layer']
                outputs = regresser(next_to_last_layer_features.float()).squeeze().cpu().numpy()

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
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="LongClip_1_layers_0.01_optim_AdamW_N100.csv")
    parser.add_argument("--N"          , '-n', type=int, help="Size of the training N+N vectors", default=100)
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--model_type"     , '-t', type=str, help="Version of the model to be tested", default='original')
    args = vars(parser.parse_args())
    
    table = test(args['in_csv'], args['device'], args['N'], args['model_type'])

    table.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions/"+args['out_csv']), index=False)  # save the results as csv file
    
    print("Testing completed")
    
    # Define the parameters to pass
    params = ["--in_csv", "data/commercial_tools.csv", "--out_csv", f"predictions/{args['out_csv']}", "--metrics", "auc", "--save_tab", f"performances/{args['out_csv']}"]

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