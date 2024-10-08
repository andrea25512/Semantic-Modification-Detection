import torch
import os
import pandas
import numpy as np
import tqdm
from PIL import Image
import torch.nn as nn

from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize
from utils.fusion import apply_fusion
from transformers import CLIPModel

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

def runnig_tests(input_csv, device, batch_size = 10):
    table = pandas.read_csv(input_csv)[['filename',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model = model.to(device).eval()
    model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))

    classifier = nn.Linear(1024, 1)
    classifier.load_state_dict(torch.load('weights/linear_SVM/SVM.pt'))
    classifier.to(device).eval()

    transform = list()
    print('input resize:', 'Clip224', flush=True)
    transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
    transform.append(CenterCrop((224, 224)))
    
    transform.append(make_normalize("clip"))
    transform = Compose(transform)
    print(flush=True)

    ### test
    with torch.no_grad():
        print(flush=True)
        
        print("Running the Tests")
        batch_img = []
        batch_id = list()
        last_index = table.index[-1]
        for index in tqdm.tqdm(table.index, total=len(table)):
            filename = os.path.join(rootdataset, table.loc[index, 'filename'])
            batch_img.append(transform(Image.open(filename).convert('RGB')))
            batch_id.append(index)

            if (len(batch_id) >= batch_size) or (index==last_index):
                batch_img = torch.stack(batch_img, 0)

                _ = model.get_image_features(pixel_values=batch_img.clone().to(device))
                
                next_to_last_layer_features = activations['next_to_last_layer']

                outputs = classifier(next_to_last_layer_features).squeeze()

                for ii, logit in zip(batch_id, outputs.cpu().numpy()):
                    table.loc[ii, 'clip'] = logit

                batch_img = []
                batch_id = list()
    print(table)
    return table


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images")
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="./results.csv")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="./weights")
    parser.add_argument("--models"     , '-m', type=str, help="List of models to test", default='clipdet_latent10k_plus,Corvi2023')
    parser.add_argument("--fusion"     , '-f', type=str, help="Fusion function", default='soft_or_prob')
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    args = vars(parser.parse_args())
    
    if args['models'] is None:
        args['models'] = os.listdir(args['weights_dir'])
    else:
        args['models'] = args['models'].split(',')
    
    table = runnig_tests(args['in_csv'], args['device'])
    #if args['fusion'] is not None:
    #    table['fusion'] = apply_fusion(table[args['models']].values, args['fusion'], axis=-1)
    
    output_csv = args['out_csv']
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    table.to_csv(output_csv, index=False)  # save the results as csv file
    
