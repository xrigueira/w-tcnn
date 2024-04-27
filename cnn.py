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
def train(dataloader, model, loss_function, optimizer, device, df_training, epoch):

    size = len(dataloader.dataset)
    model.train()
    training_loss = [] # For plotting purposes
    for i, batch in enumerate(dataloader):
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)

        # Zero out gradients for every batch
        optimizer.zero_grad()

        # Compute prediction error
        pred = model(src=src)
        pred = pred.to(device)
        loss = loss_function(pred, tgt)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Save results for plotting
        training_loss.append(loss.item())
        epoch_train_loss = np.mean(training_loss)
        df_training.loc[epoch] = [epoch, epoch_train_loss]

        # if i % 20 == 0:
        #     print('Current batch', i)
        #     loss, current = loss.item(), (i + 1) * len(src)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define the validation step
def val(dataloader, model, loss_function, device, df_validation, epoch):
    
    num_batches = len(dataloader)
    model.eval()
    validation_loss = [] # For plotting purposes
    with torch.no_grad():
        for batch in dataloader:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            
            pred = model(src=src)
            pred = pred.to(device)
            loss = loss_function(pred, tgt)

            # Save results for plotting
            validation_loss.append(loss.item())
            epoch_val_loss = np.mean(validation_loss)
            df_validation.loc[epoch] = [epoch, epoch_val_loss]
    
    loss /= num_batches
    # print(f"Avg test loss: {loss:>8f}")
    return epoch_val_loss

# Define the test function
def test(dataloader, model, device):
    
    # Get ground truth
    tgt_truth = torch.zeros(len(dataloader.dataset), 1)
    for i, (src, tgt) in enumerate(dataloader):
        tgt_truth[i] = tgt
    
    # Define tensor to store the predictions
    tgt_hat = torch.zeros(len(dataloader.dataset), 1, device=device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            src, tgt = sample
            src, tgt = src.to(device), tgt.to(device)
            
            pred = model(src=src)
            pred = pred.to(device)
            
            tgt_hat[i] = pred
    
    # Pass the tensors to the CPU
    tgt_truth = tgt_truth.cpu().numpy()
    tgt_hat = tgt_hat.cpu().numpy()

    return tgt_truth, tgt_hat


if __name__ == '__main__':
    
    # Define seed
    torch.manual_seed(0)

    station = 901

    # Define run number
    run = 1

    # Hyperparameters for the model
    batch_size = 64
    validation_size = 0.125
    variables = [f'ammonium_{station}', f'conductivity_{station}', 
                f'dissolved_oxygen_{station}', f'pH_{station}', 
                f'precipitation_{station}', f'turbidity_{station}',
                f'water_temperature_{station}', 'label']
    input_variables = variables
    timestamp_col_name = "date"
    
    # Only use data from this date and onwards
    cutoff_date = datetime.datetime(2005, 1, 1)

    n_variables = len(variables) - 1 # Exclude the label
    channels = 1
    n_classes = 2 # Normal and anomaly
    window_size = 96 # Used to slice data into sub-sequences
    step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step

    # Run parameters
    lr = 0.001
    epochs = 2

    # Get device
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using {device} device')

    # Read data
    data = utils.read_data(timestamp_col_name=timestamp_col_name)

    # Extract train and val data for nomralization purposes
    training_val_lower_bound = datetime.datetime(2005, 1, 1)
    training_val_upper_bound = datetime.datetime(2017, 12, 31)

    # Extract train/validation data
    training_val_data = data[(training_val_lower_bound >= data.index) & (data.index <= training_val_upper_bound)]
    
    # Calculate the percentage of data in the training_val_data subset
    total_data_range = (data.index.max() - data.index.min()).days
    training_val_range = (training_val_upper_bound - training_val_lower_bound).days
    train_val_percentage = (training_val_range / total_data_range)

    # Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chuncks
    data_indices = utils.get_indices(data=data, window_size=window_size, step_size=step_size)

    # Divide train data into train and validation data with a 8:1 ratio using the indices.
    # This is done this way because we need 4 years of data to create the summarized nonuniform timesteps,
    # what limits the size of the validation set. However, with this implementation, we use data from the
    # traning part to build the summarized nonuniform timesteps for the validation set. For example, if
    # we use the current validation size, the set would have less than four years of data and would not
    # be able to create the summarized nonuniform timesteps.
    training_indices = data_indices[:round(len(data_indices) * train_val_percentage)]
    validation_indices = training_indices[-round(len(training_indices) * validation_size):]
    testing_indices = data_indices[-round(len(data_indices) * (1-train_val_percentage)):]

    # Make instance of the custom dataset class
    training_data = ds.CNNDataset(data=torch.tensor(data[input_variables].values).float(), indices=training_indices)
    validation_data = ds.CNNDataset(data=torch.tensor(data[input_variables].values).float(), indices=validation_indices)
    testing_data = ds.CNNDataset(data=torch.tensor(data[input_variables].values).float(), indices=testing_indices)

    # Set up dataloaders
    training_val_data = training_data + validation_data # For testing puporses
    training_data = DataLoader(training_data, batch_size, shuffle=False)
    validation_data = DataLoader(validation_data, batch_size, shuffle=False)
    testing_data = DataLoader(testing_data, batch_size=1, shuffle=False)
    training_val_data = DataLoader(training_val_data, batch_size=1, shuffle=False) # For testing puporses

    # Instantiate the transformer model and send it to device
    model = cnn.UNet(n_variables=n_variables, channels=channels, n_classes=n_classes).to(device)

    # Print model and number of parameters
    print('Defined model:\n', model)
    utils.count_parameters(model)

    # Define optimizer and loss function
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Instantiate early stopping
    early_stopping = utils.EarlyStopping(patience=30, verbose=True)

    # Update model in the training process and test it
    start_time = time.time()
    df_training = pd.DataFrame(columns=('epoch', 'loss_train'))
    df_validation = pd.DataFrame(columns=('epoch', 'loss_val'))
    for t in range(epochs): # epochs is defined in the hyperparameters section above
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_data, model, loss_function, optimizer, device, df_training, epoch=t)
        epoch_val_loss = val(validation_data, model, loss_function, device, df_validation, epoch=t)

        # Early stopping
        early_stopping(epoch_val_loss, model, path='checkpoints')
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    print("Done! ---Execution time: %s seconds ---" % (time.time() - start_time))

    # # Save the model
    # torch.save(model, "results/models/model.pth")
    # print("Saved PyTorch entire model to models/model.pth")

    # # Load the model
    # model = torch.load("results/models/model.pth").to(device)
    # print('Loaded PyTorch model from models/model.pth')

    # Inference
    tgt_truth_train_val, tgt_hat_train_val = test(training_val_data, model, 'train_val', device)
    tgt_truth_test, tgt_hat_test = test(testing_data, model, 'test', device)

    # # Save results
    # utils.logger(run=run, batches=batch_size, d_model=d_model, n_heads=n_heads,
    #             encoder_layers=n_encoder_layers, decoder_layers=n_decoder_layers,
    #             dim_ll_encoder=in_features_encoder_linear_layer, dim_ll_decoder=in_features_decoder_linear_layer,
    #             lr=lr, epochs=epochs)

    # Plot testing results
    utils.plots_cnn(tgt_truth_train_val, tgt_hat_train_val, station=station, phase='train_val', run=run)
    utils.plots_cnn(tgt_truth_test, tgt_hat_test, station=station, phase='test', run=run)

    # Metrics
    utils.metrics(tgt_truth_train_val, tgt_hat_train_val, 'train_val')
    utils.metrics(tgt_truth_test, tgt_hat_test, 'test')