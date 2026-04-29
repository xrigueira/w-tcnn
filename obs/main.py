import torch
import pickle
import numpy as np
import pandas as pd

import utils

def predict_unet(data, results_transformer, model_unet):
    
    tgts_truth = []
    y_hats = []
    
    # Perform prediction
    model_unet.eval()
    with torch.no_grad():

        for key in results_transformer.keys():

            # Load data
            y_hat_transformer = torch.tensor(results_transformer[key]['y_hat'], dtype=torch.float32)
            y_hat_transformer = y_hat_transformer.unsqueeze(0).unsqueeze(0)
            
            dates = pd.date_range(start=key, end=key + pd.Timedelta(hours=1), freq='15min')[:-1]
            tgt_unet = torch.tensor(data.loc[dates, 'label'].values.mean(axis=0))
            
            tgt_unet = (tgt_unet > 0.5).float()  # Convert to binary
        
            # Send data to device
            y_hat_transformer, tgt_unet = y_hat_transformer.to(device), tgt_unet.to(device)

            # Perform prediction
            y_hat_unet = model_unet(src=y_hat_transformer)

            # Convert to binary
            y_hat_unet = (y_hat_unet > 0.5).float()

            # Store results as Python floats
            tgts_truth.append(tgt_unet.cpu().item())  # Convert scalar tensor to float
            y_hats.append(y_hat_unet.cpu().squeeze().item())  # Convert single-value tensor to float
        
    return tgts_truth, y_hats

def predict_rf(data, results_transformer, rf_model):
    
    tgts_truth = []
    y_hats = []

    for key in results_transformer.keys():
        # Load data
        y_hat_transformer = results_transformer[key]['y_hat'].flatten()  # Flatten to 1D array
        
        dates = pd.date_range(start=key, end=key + pd.Timedelta(hours=1), freq='15min')[:-1]
        tgt_rf = data.loc[dates, 'label'].values.mean()  # Get the mean label for the hour
        tgt_rf = int(tgt_rf > 0.5)  # Convert to binary
        
        # Perform prediction
        y_hat_rf = rf_model.predict([y_hat_transformer])[0]  # Predict and get the first (and only) result

        # Store results
        tgts_truth.append(tgt_rf)
        y_hats.append(y_hat_rf)
    
    return tgts_truth, y_hats

run = 0
phase = 'test'

# Get device
device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using {device} device')

# Load data
data = utils.read_data(timestamp_col_name='date')

# Load transformer results
results_transformer = np.load(f'results/run_t_{run}/results_{phase}.npy', allow_pickle=True, fix_imports=True).item()  # Convert back to dict

# Load UNet model
# model_unet = torch.load(f'results/run_u_{run}/unet_model.pth').to(device)

# tgts_truth_unet, y_hats_unet = predict_unet(data, results_transformer, model_unet)
# np.save('results/tgts_truth_unet.npy', tgts_truth_unet, allow_pickle=True, fix_imports=True)
# np.save('results/y_hats_unet.npy', y_hats_unet, allow_pickle=True, fix_imports=True)

# Load RF model
rf_model = pickle.load(open('models/rf_model.sav', 'rb'))

tgts_truth_rf, y_hats_rf = predict_rf(data, results_transformer, rf_model)
np.save('results/tgts_truth_rf.npy', tgts_truth_rf, allow_pickle=True, fix_imports=True)
np.save('results/y_hats_rf.npy', y_hats_rf, allow_pickle=True, fix_imports=True)

# # Print confusion matrix
# from sklearn.metrics import confusion_matrix
# cm_unet = confusion_matrix(tgts_truth_unet, y_hats_unet)
# print(cm_unet)

# Print confusion matrix
from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(tgts_truth_rf, y_hats_rf)
print(cm_rf)