## Test Code
Before using the code, download the weights:

```
git lfs pull
```

The `main.py` script requires as input a CSV file with the list of images to analyze.
The input CSV file must have a 'filename' column with the path to the images.
The code outputs a CSV file with the LLR score for each image.
If LLR>0, the image is detected as synthetic.

The `compute_metrics.py` script can be used to evaluate metrics.
In this case, the input CSV file must also include the 'typ' column with a value equal to 'real' for real images.


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

The test can be executed as follows:

```
python main.py --in_csv /path/input/csv --out_csv /path/output/csv --device 'cuda:0'
```

To get the results on Commercial Tools generators:
1) Firstly, download the synthbuster dataset using the following command:
```
cd data; bash synthbuster_download.sh; cd ..
```

2) Then, run the `new_main.py` script as follows: 
```
python new_main.py --in_csv data/commercial_tools.csv --out_csv out.csv --device 'cuda:0'
```

3) Finally, calculate the AUC metrics:
```
python compute_metrics.py --in_csv data/commercial_tools.csv --out_csv out.csv --metrics auc --save_tab auc_table.csv
```