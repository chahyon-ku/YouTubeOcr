import data
from data import CharDataset
import torch

data.generate_data()
dataset = CharDataset()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, sampler=None, num_workers=0, pin_memory=True)

