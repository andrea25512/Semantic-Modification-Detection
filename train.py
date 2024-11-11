import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.dataset_augmentations import createDataset
from transformers import CLIPModel
from networks.shallow import TwoRegressor, ThreeRegressor
from torch.utils.tensorboard import SummaryWriter

activations = {}
seed = 42
hidde_size = 512
torch.manual_seed(seed)

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

def train(input_folder, weights_dir, N, device, layers, learning_rate):
    print(input_folder)
    dataset = createDataset(input_folder, device, train=True)
    train_dataset, test_dataset = random_split(dataset, [N, len(dataset) - N])
    
    print("Training images: ",len(train_dataset)," - Test images: ",len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model = model.to(device).eval()
    model.vision_model.post_layernorm.register_forward_hook(get_activation('next_to_last_layer'))

    regresser = None
    if(layers == 1):
        regresser = nn.Linear(1024, 1).to(device)
    elif(layers == 2):
        regresser = TwoRegressor(hidde_size).to(device)
    elif(layers == 3):
        regresser = ThreeRegressor(hidde_size).to(device)
    else:
        print("Only supported from 1 to 3 layers")
        exit()

    optimizer = optim.AdamW(regresser.parameters(), lr=learning_rate)

    name = None
    if(layers == 1):
        name = '1_layers_'+str(learning_rate)+'_optim_'+str(type(optimizer).__name__)+'_N'+str(N)
    else:
        name = f"{layers}_layers_{hidde_size}_ReLU_{learning_rate}_optim_{type(optimizer).__name__}_N{N}_hinge"
    print("Output name: ",name)
    writer = SummaryWriter('runs/'+name)

    print(flush=True)

    ### training
    print("Running the Training")

    for idx, (real_images, synthetic_images, tot_real_entries, tot_fake_entries) in enumerate(tqdm(train_loader)):
        total_images = torch.cat((real_images, synthetic_images), dim=0)
        total_images = total_images.view(-1, 3, 224, 224)
  
        _ = model.get_image_features(pixel_values=total_images.clone().to(device)).cpu()
        
        next_to_last_layer_features = activations['next_to_last_layer'].cpu()

        real_images_labels = torch.zeros(real_images.size(0)*real_images.size(1))
        synthetic_images_labels = torch.ones(synthetic_images.size(0)*synthetic_images.size(1))
        
        labels = torch.cat((real_images_labels, synthetic_images_labels), dim=0).to(device)
        labels = 2 * labels - 1  # Convert labels from 0/1 to -1/+1

        outputs = regresser(next_to_last_layer_features.to(device)).squeeze()

        loss = torch.mean(torch.clamp(1 - outputs * labels, min=0))
        writer.add_scalar('training loss', loss, idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # save
    writer.close()
    torch.save(regresser.state_dict(), weights_dir+name+'.pt')


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder"     , '-i', type=str, help="The path of the dataset folder with the folders of the images' origin", default="../../Baseline/Semantic-Modification-Detection/data/synthbuster")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="weights/shallow/")
    parser.add_argument("--N"          , '-n', type=int, help="Size of the training N+N vectors", default=100)
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--layers"     , '-l', type=int, help="Number of layers of the regressor", default=3)
    parser.add_argument("--learning_rate"     , '-r', type=float, help="Learning rate of the optimizer", default=0.0005)
    args = vars(parser.parse_args())
    
    train(args['in_folder'], args['weights_dir'], args['N'], args['device'], args['layers'], args['learning_rate'])
    
    print("Training completed")