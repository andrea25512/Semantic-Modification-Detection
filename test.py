import torch
import os
import pandas
from tqdm import tqdm
import torch.nn as nn
import yaml
from networks import create_architecture, load_weights
from torch.utils.data import DataLoader, random_split
from transformers import CLIPModel
from networks.shallow import TwoRegressor, ThreeRegressor
from utils.dataset_augmentations import createDataset

activations = {}
seed = 42
hidden_size = 512
torch.manual_seed(seed)

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']

def test(input_folder, device, N, model_type):
    print(input_folder)
    dataset = createDataset(input_folder, device, train=False)
    train_dataset, test_dataset = random_split(dataset, [N, len(dataset) - N])

    print("Training images: ",len(train_dataset)," - Test images: ",len(test_dataset))

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    global location
    model = None
    regresser = None
    if(model_type[0] == '1'):
        print("Linear SVM")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
        regresser = nn.Linear(1024, 1).to(device)
    elif(model_type[0] == '2'):
        print("Two layers shallow network")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
        regresser = TwoRegressor(hidden_size).to(device)
        regresser.load_state_dict(torch.load("weights/shallow/"+model_type+".pt"))
    elif(model_type[0] == '3'):
        print("Three layers shallow network")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))
        regresser = ThreeRegressor(hidden_size).to(device)
        regresser.load_state_dict(torch.load("weights/shallow/"+model_type+".pt"))
    elif(model_type == "original"):
        print("Original paper network")
        _, model_path, arch, norm_type, patch_size = get_config("clipdet_latent10k", weights_dir="../../Baseline/Semantic-Modification-Detection/weights")
        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()
        location = "original"
    else:
        print("Wrong model selection")
        exit()

    output = pandas.DataFrame(columns=['filename','clip','aug'])

    ### training
    print("Running the Testing")
    with torch.no_grad():
        for index, (real_images, synthetic_images, tot_real_entries, tot_fake_entries) in enumerate(tqdm(test_loader)):
            total_images = torch.cat((real_images.squeeze(), synthetic_images.squeeze()), dim=0)

            labels = torch.tensor(torch.cat((torch.zeros(real_images.shape[1]), torch.ones(synthetic_images.shape[1])), dim=0), dtype=torch.float32).to(device)
            labels = 2 * labels - 1  # Convert labels from 0/1 to -1/+1

            outputs = None
            if(model_type == "original"):
                outputs = model(total_images.clone().to(device)).cpu().squeeze().tolist()
            else:
                _ = model.get_image_features(pixel_values=total_images.clone().to(device)).cpu()
                next_to_last_layer_features = activations['next_to_last_layer'].cpu()
                outputs = regresser(next_to_last_layer_features.to(device)).squeeze().cpu().numpy()

            for idx, (ii, logit) in enumerate(zip(tot_real_entries + tot_fake_entries, outputs)):
                aug_idx = idx % len(dataset.captions)
                aug_name = dataset.captions[aug_idx]  # Select the correct augmentation name
                
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
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images", default="../../Baseline/Semantic-Modification-Detection/data/synthbuster")
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output folder for the csv file", default="predictions/")
    parser.add_argument("--N"          , '-n', type=int, help="Size of the training N+N vectors", default=100)
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--model_type"     , '-t', type=str, help="Version of the model to be tested", default='original')
    args = vars(parser.parse_args())
    
    table = test(args['in_csv'], args['device'], args['N'], args['model_type'])

    os.makedirs(os.path.dirname(os.path.abspath(args['out_csv']+args['model_type']+".csv")), exist_ok=True)
    table.to_csv(args['out_csv']+args['model_type']+".csv", index=False)  # save the results as csv file
    
    print("Testing completed")