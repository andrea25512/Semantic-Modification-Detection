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
import random

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

def test(input_csv, device, N, model_type, is_forlab):
    # making the script runnable from anywhere
    torch.manual_seed(seed)
    random.seed(seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rootdataset = os.path.join(script_dir,input_csv)
    RAISE_data_folder = os.path.dirname(os.path.abspath(rootdataset))
    # since no augmentations are needed, we can read the images directly. Thus here we load the image dictionary
    table = pandas.read_csv(rootdataset)
    # sampled N real images from the dictionary. The semi-random sampling is done on the last 1000 entries since the image dictionary is constructed as [synthetic1, synthetic2, synthetic3, synthetic4, real], where each subset is of size 1000 (5000 entries total)
    train_real_sample_indices = table.iloc[-1000:].sample(n=N, random_state=seed).index
    # we get the test dataset samples by simply subtrcting from the original one the sample of the training set/ The training set is replicated thanks to the fixed seed
    test_real_sample_indices = table.iloc[-1000:].index.difference(train_real_sample_indices)
    RAISE_real_samples = table.loc[test_real_sample_indices].reset_index(drop=True)

    # if during training the StableDiffusion datast was utilized (alternative to synthetic), we need to test the distribution shift in another real dataset, in this case the FORLAB was selected
    if(is_forlab):
        FORLAB_real_samples = [f for f in os.listdir("/media/mmlab/Volume2/TrueFake/PreSocial/Real/FORLAB") if os.path.isfile(os.path.join("/media/mmlab/Volume2/TrueFake/PreSocial/Real/FORLAB", f))]
        # a semi-random shuffle of the selected N indexes is applied, otherwise we cannot test the last images if not with low N number
        random.shuffle(FORLAB_real_samples)

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
        transform = list()
        transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
        transform.append(CenterCrop((224, 224)))
        transform.append(make_normalize("clip"))
        transform = Compose(transform)

    # here the CSV from the predited logits is declared
    output = pandas.DataFrame(columns=['filename','clip'])

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("Running the Testing")
    # batch size is fized to one since we load directly the images without creating a dataset + dataloader instances
    batch_img = []
    with torch.no_grad():
        for index in tqdm.tqdm(range(len(RAISE_real_samples)), dynamic_ncols=True, file=sys.stderr):
            # the real file name under the real_RAISE_1k OR forlab is extracted.
            if(is_forlab):
                real_filename = os.path.join("/media/mmlab/Volume2/TrueFake/PreSocial/Real/FORLAB", FORLAB_real_samples[index])
            else:
                real_filename = os.path.join(RAISE_data_folder, RAISE_real_samples.loc[index, 'filename'])

            # all the four synthetic variants are loaded
            synthetic_filename1 = os.path.join(RAISE_data_folder, "synthbuster/dalle2/"+RAISE_real_samples.loc[index, 'filename'].split('/')[-1])
            synthetic_filename2 = os.path.join(RAISE_data_folder, "synthbuster/dalle3/"+RAISE_real_samples.loc[index, 'filename'].split('/')[-1])
            synthetic_filename3 = os.path.join(RAISE_data_folder, "synthbuster/firefly/"+RAISE_real_samples.loc[index, 'filename'].split('/')[-1])
            synthetic_filename4 = os.path.join(RAISE_data_folder, "synthbuster/midjourney-v5/"+RAISE_real_samples.loc[index, 'filename'].split('/')[-1])
            
            # the real and synthetic images are put in the same batch, as well to the labels 
            batch_img.append(transform(Image.open(real_filename).convert('RGB')))
            batch_img.append(transform(Image.open(synthetic_filename1).convert('RGB')))
            batch_img.append(transform(Image.open(synthetic_filename2).convert('RGB')))
            batch_img.append(transform(Image.open(synthetic_filename3).convert('RGB')))
            batch_img.append(transform(Image.open(synthetic_filename4).convert('RGB')))

            # here the LLM2CLIP preprocesser gives back the images in another more accessible format, thus the extraction via "item['pixel_values'][0]" is not needed
            if("LLM2CLIP" not in model_type):
                batch_img = torch.stack(batch_img, 0)
            else:
                batch_img = [item['pixel_values'][0] for item in batch_img]
                batch_img = torch.stack([torch.tensor(arr) for arr in batch_img])

            outputs = None
            if(model_type == "original"):
                # call to the model in the Verdoliva paper 
                outputs = model(batch_img.clone().to(device)).cpu().squeeze().tolist()
            elif("LongClip" in model_type):
                # call to the LongCLIP model 
                _ = model.encode_image(batch_img.clone().to(device))
                next_to_last_layer_features = activations['next_to_last_layer']
                outputs = regresser(next_to_last_layer_features.float()).squeeze().cpu().numpy()
            else:
                # call to the CLIP and LLM2CLIP models
                _ = model.get_image_features(pixel_values=batch_img.clone().to(device))
                next_to_last_layer_features = activations['next_to_last_layer']
                outputs = regresser(next_to_last_layer_features.float()).squeeze().cpu().numpy()

            # here the logits are paired with the image name, in order to do in another file the metric calculation. The table will be saved to a CSV file
            # even if we use the FORLAB dataset instead of the real_RAISE_1k, we still save the real images under the real_RAISE_1k name in order to have back compatibility with the metrics computation
            for ii, logit in zip([RAISE_real_samples.loc[index, 'filename'],"synthbuster/dalle2/"+RAISE_real_samples.loc[index, 'filename'].split('/')[-1],"synthbuster/dalle3/"+RAISE_real_samples.loc[index, 'filename'].split('/')[-1],"synthbuster/firefly/"+RAISE_real_samples.loc[index, 'filename'].split('/')[-1],"synthbuster/midjourney-v5/"+RAISE_real_samples.loc[index, 'filename'].split('/')[-1]], outputs):
                entry = pandas.DataFrame.from_dict({
                    "filename": [ii],
                    "clip":  [logit]
                })
                if output.empty:
                    output = entry
                else:
                    output = pandas.concat([output, entry], ignore_index=True)

            batch_img = []

    # return the table
    return output

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images", default="data/commercial_tools.csv")
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="LongClip_1_layers_0.01_optim_AdamW_N100.csv")
    parser.add_argument("--N"          , '-n', type=int, help="Size of the training N+N vectors", default=100)
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--model_type"     , '-m', type=str, help="Version of the model to be tested", default='original')
    parser.add_argument("--forlab"     , '-f', action="store_true", help="Enable testing with FORLAB as the synthetic images (default: False value)")
    args = vars(parser.parse_args())
    
    table = test(args['in_csv'], args['device'], args['N'], args['model_type'], args['forlab'])

    table.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions/"+args['out_csv']), index=False)  # save the results as csv file
    
    print("Testing completed")
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # here we continue the pipeline by calling automatically the metrics computation on the CSV

    # we reuse most of the parameters that this code recieved from launch, but most importantly we pass the newely constructed CSV name
    params = ["--in_csv", "data/commercial_tools.csv", "--out_csv", f"predictions/{args['out_csv']}", "--metrics", "auc", "--save_tab", f"performances/{args['out_csv']}"]

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