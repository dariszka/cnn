import os
from torch.utils.data import DataLoader, Subset
from dataset import ImagesDataset
import numpy as np

# from 05_data_loading.py 
rng = np.random.default_rng(seed=333)

path = os.path.abspath('training_data')
batch_size = 64
train_split = 0.7
val_split = 0.2
test_split = 0.1

ds = ImagesDataset(path)

n_samples = len(ds)
shuffled_indices = rng.permutation(n_samples)

test_set_indices = shuffled_indices[:int(n_samples * test_split)]
validation_set_indices = shuffled_indices[len(test_set_indices):len(test_set_indices)+int(n_samples*val_split)]
training_set_indices = shuffled_indices[len(validation_set_indices):]

test_set = Subset(ds, indices=test_set_indices)
validation_set = Subset(ds, indices=validation_set_indices)
training_set = Subset(ds, indices=training_set_indices)

test_loader = DataLoader(
    test_set, 
    shuffle=False, 
    batch_size=1  
)
validation_loader = DataLoader(
    validation_set,  
    shuffle=False,  
    batch_size=batch_size  
)
training_loader = DataLoader(
    training_set, 
    shuffle=True,  
    batch_size=batch_size  
)
