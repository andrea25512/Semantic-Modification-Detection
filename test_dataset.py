from utils.dataset_augmentations import createDataset

path_to_dataset = "../../Baseline/Semantic-Modification-Detection/data/synthbuster"

dataset = createDataset(path_to_dataset, "cuda:1", train=False, debug=False)

real_images, tot_synthetic_images, tot_real_entries, tot_fake_entries = dataset[0]
print(real_images.shape)
print(tot_synthetic_images.shape)