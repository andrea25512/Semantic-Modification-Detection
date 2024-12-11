## Training
The `train.py` is utilized to train different possible configurations of classifiers.
First a selection of the possible feature extractor can be performed

	* CLIP
 	* LongCLIP
 	* LLM2CLIP

Then it can also be selected the lenght of the shallow model, from 1 to 5 layers, other than the common training parameters like Learning Rate and so on
The code will output a .pt file with the weights of the trained shallow network

## Testing
The `test.py` is utilized to test different possible configurations of classifiers, as it accepts a .pt file with a specific nomenclature. 
The code will output a .csv file with the predictions of the test set

## Metrics
The `compute_metrics.py` script can be used to evaluate metrics, and it accepts a .csv file with evaluations.
It will generate a AUC metric in its standard configuration.

### Script 
In order to use the script the follwing packages should be installed:

	* tqdm
	* scikit-learn
	* pillow
	* yaml
	* pandas
	* torchvision
	* torch
	* timm>=0.9.10
	* huggingface-hub>=0.23.0
	* open_clip_torch

---

The train can be executed as follows:

```
python train.py --in_csv data/commercial_tools.csv --weights_dir weights/shallow/ --N 100 --device 'cuda:0' --layers 3 --learning_rate 0.005 --version 'LLM2CLIP' --stable_diffusion --augmentations
```
The `--augmentations` parameter can be included to remove the augmentations application, a fallback to the `Baseline` branch behaviour.

The `--stable_diffusion` parameter is utilized for selecting a training on:

	* real_RAISE_1k dataset for REAL images
 	* StableDiffusion dataset for SYNTHETIC images 

And a testing on:
	
 	* FORLAB dataset for REAL images
  	* dalle2, dalle3, firefly, midjourney-v5 datasets for SYNTHETIC images

Thus permitting to evaluate a shift in the distribution bethween the training and testing data.

Not including this parameter will instead fallback the configuration of the training to the old configuration: 

	* real_RAISE_1k dataset for REAL images
 	* dalle2, dalle3, firefly, midjourney-v5 datasets for SYNTHETIC images 

And a testing on:

	* real_RAISE_1k dataset for REAL images
 	* dalle2, dalle3, firefly, midjourney-v5 datasets for SYNTHETIC images 

Thus, even if there is a split in the dataset for training and testing data, the distribution shift of changing datasets is not present this way

--- 

The test can be executed as follows:

```
python test.py --in_csv data/commercial_tools.csv --out_csv predictions/LongClip_1_layers_0.01_optim_AdamW_N100.csv --N 100 --device 'cuda:0' --model_type 'original' --forlab --augmentations
```
The `--forlab` parameter must be included if the training was done with the `--stable_diffusion` parameter set

---

The metrics computation can be executed as follows:

```
python compute_metrics.py --in_csv data/commercial_tools.csv --out_csv predictions/LongClip_1_layers_0.01_optim_AdamW_N100.csv --metrics 'auc' --save_tab 'performances/LongClip_1_layers_0.01_optim_AdamW_N100.csv'
```

This additional metrics computation makes possible to generate AUC accuracy focusing on each of the possible augmentations, instead of the different fake datasets like in the original `compute_metrics.py` code

```
python augmentation_wise_metrics.py --in_csv data/commercial_tools.csv --out_csv predictions/LongClip_1_layers_0.01_optim_AdamW_N100.csv --metrics 'auc' --save_tab 'performances/LongClip_1_layers_0.01_optim_AdamW_N100.csv'
```

## Other versions
Another version of this code, that does not apply augmentations during training and testing, can be found in the `Baseline` branch

Another version of this code, that instead of detecting if an image is real or fake, detects how mutch or where the inpainting was applied can be found in the `Inpainting` branch