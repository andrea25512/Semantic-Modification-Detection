import os
import pandas as pd
import numpy as np
from sklearn import metrics 

dict_metrics = {
    'auc' : lambda label, score: metrics.roc_auc_score(label,  score),
    'acc' : lambda label, score: metrics.balanced_accuracy_score(label, score > 0),
}

def compute_metrics(input_csv, output_csv, metrics_fun):
    # Read the input CSVs
    input_table = pd.read_csv(input_csv)
    output_table = pd.read_csv(output_csv)
    
    # Merge tables on 'filename'
    table = pd.merge(input_table, output_table, on='filename')
    
    # Ensure 'clip' is in the table
    assert 'clip' in table.columns, "'clip' column not found in the data."
    
    # Assign labels: 0 for real images, 1 for synthetic
    table['label'] = (table['typ'] != 'real').astype(int)
    
    # Get list of synthetic types (exclude 'real') and augmentations
    list_typs = sorted(set(table['typ']) - {'real'})
    list_augs = sorted(table['aug'].unique())
    
    # Initialize a DataFrame to store results
    tab_metrics = pd.DataFrame(index=list_augs, columns=list_typs)
    tab_metrics.loc[:, :] = np.nan
    
    # Compute metrics for each synthetic type and augmentation
    for typ in list_typs:
        for aug in list_augs:
            # Filter data for current type and augmentation, including real images
            tab_typ = table[(table['typ'].isin(['real', typ])) & (table['aug'] == aug)]
            
            if tab_typ.empty:
                continue  # Skip if no data for this combination
            
            # Get scores and labels
            scores = tab_typ['clip'].values
            labels = tab_typ['label'].values
            
            # Check if there are both real and synthetic images
            if len(np.unique(labels)) < 2:
                continue  # Cannot compute AUC with only one class
            
            # Compute the metric
            metric_value = metrics_fun(labels, scores)
            
            # Store the metric value
            tab_metrics.loc[aug, typ] = metric_value
    
    # Calculate the average across types for each augmentation
    tab_metrics['AVG'] = tab_metrics.mean(axis=1)

    # Calculate the average across augmentations for each synthetic type (column mean)
    tab_metrics.loc['Mean'] = tab_metrics.mean(axis=0)
    
    return tab_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", '-i', type=str, help="The path of the input CSV file with the list of images", default="../../Baseline/Semantic-Modification-Detection/data/commercial_tools.csv")
    parser.add_argument("--out_csv", '-o', type=str, help="The path of the output CSV file", default="predictions/3_layers_512_ReLU_0.0005_optim_AdamW_N100_hinge.csv")
    parser.add_argument("--metrics", '-w', type=str, help="Type of metrics ('auc' or 'acc')", default="auc")
    parser.add_argument("--save_tab", '-t', type=str, help="The path of the metrics CSV file", default="tmp.csv")
    args = vars(parser.parse_args())
    
    tab_metrics = compute_metrics(args['in_csv'], args['out_csv'], dict_metrics[args['metrics']])
    tab_metrics.index.name = 'augmentation'
    
    # Display the results
    print(tab_metrics.to_string(float_format=lambda x: '%5.3f' % x))
    
    if args['save_tab'] is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args['save_tab'])), exist_ok=True)
        tab_metrics.to_csv(args['save_tab'])
