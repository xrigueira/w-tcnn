import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
import dataset as ds
import models.cnn as cnn

# Define the training step
def train():
    pass

# Define the validation step
def val():
    pass

# Define the main function
def main():
    pass

if __name__ == '__main__':
    
    # Define seed
    torch.manual_seed(0)

    station = 901

    # Define run number
    run = 1

    # Hyperparameters for the model
    # Continue defining the hyperparameters
    # batch_size = 64
    variables = [f'ammonium_{station}', f'conductivity_{station}', 
                f'dissolved_oxygen_{station}', f'pH_{station}', 
                f'precipitation_{station}', f'turbidity_{station}',
                f'water_temperature_{station}', 'label']
    