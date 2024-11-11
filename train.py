import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from transformers import CLIPModel
from networks.shallow import TwoRegressor, ThreeRegressor
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import createDataset
from tqdm import tqdm

activations = {}
seed = 42
hidde_size = 512
torch.manual_seed(seed)

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

def train(rootdataset, device, dataset_ratio, layers, weights_dir, learning_rate, batch_size = 16):
    dataset = createDataset(rootdataset)
    train_size = int(dataset_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_test_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    split = len(val_test_dataset) // 2
    val_dataset, test_dataset = random_split(val_test_dataset, [split, len(val_test_dataset)-split])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

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
    print(flush=True)
    delta = 0.5

    name = None
    if(layers == 1):
        name = '1_layers_'+str(learning_rate)+'_optim_'+str(type(optimizer).__name__)+'_N'+str(dataset_ratio)
    else:
        name = f"{layers}_layers_{hidde_size}_ReLU_{learning_rate}_optim_{type(optimizer).__name__}_N{dataset_ratio}_mse"
    print("Output name: ",name)
    writer = SummaryWriter('runs/'+name)

    ### training
    print("Running the Training")
    best_loss = float('inf')
    #best_model = None
    for epoch in range(1):
        regresser.train()
        for index, (inpainted_images, inpainted_ratio) in enumerate(tqdm(train_loader)):
            images = inpainted_images.view(-1, 3, 224, 224)
            
            _ = model.get_image_features(pixel_values=images.clone().to(device)).cpu()
            next_to_last_layer_features = activations['next_to_last_layer'].cpu()

            outputs = regresser(next_to_last_layer_features.to(device)).squeeze()
            loss = nn.functional.mse_loss(outputs.cpu(), inpainted_ratio.float())
            writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + index)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ### Validation phase
        regresser.eval()
        val_loss = 0.0
        with torch.no_grad():
            for index, (inpainted_images, inpainted_ratio) in enumerate(tqdm(val_loader)):
                images = inpainted_images.view(-1, 3, 224, 224)

                _ = model.get_image_features(pixel_values=images.clone().to(device)).cpu()
                next_to_last_layer_features = activations['next_to_last_layer'].cpu()

                outputs = regresser(next_to_last_layer_features.to(device)).squeeze()
                loss = nn.functional.mse_loss(outputs.cpu(), inpainted_ratio.float())
                val_loss += loss.item()

        val_loss /= len(val_loader)
        writer.add_scalar('validation loss', val_loss, epoch)
        

    # save
    writer.close()
    print("best loss: ",best_loss)
    torch.save(regresser.state_dict(), weights_dir+name+'.pt')

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the root folder containing both the inpaited and the mask images", default="/media/mmlab/Volume2/")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="weights/shallow/")
    parser.add_argument("--N"          , '-n', type=float, help="Size of the training dataset", default=0.8)
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--layers"     , '-l', type=int, help="Number of layers of the regressor", default=3)
    parser.add_argument("--learning_rate"     , '-r', type=float, help="Learning rate of the optimizer", default=0.005)
    args = vars(parser.parse_args())
    
    train(args['in_csv'], args['device'], args['N'], args['layers'], args['weights_dir'], args['learning_rate'])
    
    print("Training completed")