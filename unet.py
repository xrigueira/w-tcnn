import time
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

import utils
import dataset as ds
import models.cnn as cnn

# Define the training step
def train(dataloader, model, loss_function, optimizer, device, df_training, epoch):

    num_batches = len(dataloader)
    training_loss = 0.0 # Initialize as a scalar for cumulative loss

    # Set model to training mode
    model.train()
    
    for i, batch in enumerate(dataloader):
        src, tgt, tgt_abs = batch
        src = src.unsqueeze(0).permute(1, 0, 2, 3) # Change the shape of the tensor to (batch_size, channels, window_size, num_features)
        
        # Send data to device
        src, tgt, tgt_abs = src.to(device), tgt.to(device), tgt_abs.to(device)

        # Zero out gradients for every batch
        optimizer.zero_grad()

        # Compute prediction error
        y_hat = model(src=src)

        # Compute loss
        loss = loss_function(y_hat, tgt)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss for the epoch
        training_loss += loss.item()
    
    # Calculate the average training loss for the epoch
    epoch_train_loss = training_loss / num_batches

    # Append to the DataFrame
    df_training.loc[len(df_training)] = [epoch, epoch_train_loss]

# Define the validation step
def val(dataloader, model, loss_function, device, df_validation, epoch):
    
    num_batches = len(dataloader)
    validation_loss = 0.0 # Initialize as a scalar for cumulative loss

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            src, tgt, tgt_hat = batch
            src = src.unsqueeze(0).permute(1, 0, 2, 3) # Change the shape of the tensor to (batch_size, channels, window_size, num_features)
            
            # Send data to device
            src, tgt, tgt_hat = src.to(device), tgt.to(device), tgt_hat.to(device)
            
            # Forward pass
            y_hat = model(src=src)

            # Compute loss
            loss = loss_function(y_hat, tgt)

            # Accumulate loss for the epoch
            validation_loss += loss.item()
            
        # Calculate the average validation loss for the epoch
        epoch_val_loss = validation_loss / num_batches

        # Append to the DataFrame
        df_validation.loc[len(df_validation)] = [epoch, epoch_val_loss]

    return epoch_val_loss

# Define the test function
def test(dataloader, model, phase, device):
    
    # Get ground truth
    tgts_truth = torch.zeros(len(dataloader), device=device)
    y_hats = torch.zeros(len(dataloader), device=device)
    
    # # Define tensor to store the predictions
    # tgt_hat = torch.zeros(len(dataloader.dataset), 1, device=device)

    # Define a dictionary to store the predictions
    results = {}

    # Perform inference
    model.eval()
    with torch.no_grad():
        for i, (src, tgt, tgt_abs) in enumerate(dataloader):
            
            src = src.unsqueeze(0).permute(1, 0, 2, 3) # Change the shape of the tensor to (batch_size, channels, window_size, num_features)
            
            # Send data to device
            src, tgt, tgt_abs = src.to(device), tgt.to(device), tgt_abs.to(device)

            # Forward pass
            y_hat = model(src=src)

            # Store ground truth and prediction
            tgts_truth[i] = tgt
            y_hat[i] = y_hat

            # Save the results
            if phase != '':
                results[dates[i]] = {
                    'src': src.cpu().detach().numpy(),
                    'tgt': tgt.cpu().detach().numpy(),
                    'y_hat': y_hat.cpu().detach().numpy(),
                }
    
    # Save the dictionary to a numpy file
    if phase != '':
        np.save(f'results/run_u_{run}/results.npy', results, allow_pickle=True, fix_imports=False)
    
    # Pass the tensors to the CPU and convert to numpy arrays
    tgts_truth = tgts_truth.cpu().numpy()
    y_hat = y_hat.cpu().numpy()

    return tgts_truth, y_hat

if __name__ == '__main__':
    
    # Define seeds
    seed = 0
    random.seed(seed) # Affects the random selection of the validation set
    np.random.seed(seed) # Affects random noise added to the data

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    station = 901

    # Define run number
    run = 0

    # Hyperparameters for the model
    batch_size = 128
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
    window_size = 4 # Used to slice data into sub-sequences
    step_size = 4 # Step size, i.e. how many time steps does the moving window move at each step
    
    input_channels = 1
    channels = 2
    d_fc = 128
    n_classes = 1 # Anomaly or non anomaly

    # Run parameters
    lr = 0.001
    epochs = 1

    # Get device
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using {device} device')

    # Read data
    data = utils.read_data(timestamp_col_name=timestamp_col_name)

    # Define the training and validation bounds
    training_val_lower_bound = pd.Timestamp(2005, 1, 1)
    training_val_upper_bound = pd.Timestamp(2017, 12, 31)

    # Get the global index of the train_upper_bound to later subset the data indices
    training_val_upper_index = round(data.index.get_loc(training_val_upper_bound) / step_size)

    # Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chuncks
    data_indices = utils.get_indices(data=data, window_size=window_size, step_size=step_size)
    
    # Divide data into train and validation data with a 8:1 ratio using the indices.
    training_validation_indices = data_indices[:(training_val_upper_index)]
    training_indices = training_validation_indices[:(round(len(training_validation_indices) * (1-validation_size)))]
    validation_indices = training_validation_indices[-round(len(training_validation_indices) * validation_size):]
    testing_indices = data_indices[(training_val_upper_index):]
    
    # Get the complete range of dates for the train and validation sets
    # dates = (pd.Series(data.index.date).drop_duplicates().sort_values()).to_list()  # Remove duplicates and sort
    dates = (pd.Series(data.index.floor('h')).drop_duplicates().sort_values()).to_list()  # Remove duplicates and sort
    
    # Subset the test and validation sets
    dates_train_validation = dates[int(window_size/step_size):len(training_validation_indices)]
    dates_train = dates_train_validation[:(round(len(training_validation_indices) * (1-validation_size)))]
    dates_validation = dates_train_validation[-round(len(training_validation_indices) * validation_size):]
    
    # Get the complete range of dates for the test set
    dates_test = dates[len(training_validation_indices):]

    # Make instance of the custom dataset class
    training_data = ds.UNetDataset(data=torch.tensor(data[input_variables].values).float(), indices=training_indices)
    validation_data = ds.UNetDataset(data=torch.tensor(data[input_variables].values).float(), indices=validation_indices)
    testing_data = ds.UNetDataset(data=torch.tensor(data[input_variables].values).float(), indices=testing_indices)
    
    # Set up the train and validation dataloaders used for inference
    training_data_inference = DataLoader(training_data, batch_size=1, shuffle=False)
    validation_data_inference = DataLoader(validation_data, batch_size=1, shuffle=False)

    # Set up the dataloaders
    training_data = DataLoader(training_data, batch_size, shuffle=True)
    validation_data = DataLoader(validation_data, batch_size, shuffle=True)
    testing_data = DataLoader(testing_data, batch_size=1, shuffle=False)

    # Instantiate the cnn model and send it to device
    model = cnn.UNet(n_variables=n_variables, window_size=window_size, n_classes=n_classes,
                    input_channels=input_channels, channels=channels, d_fc=d_fc).to(device)

    # Print model and number of parameters
    print('Defined model:\n', model)
    utils.count_parameters(model)

    # Define optimizer and loss function
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Instatiate scheduler
    # scheduler = utils.LearningRateSchedulerPlateau(optimizer, patience=10, factor=0.90)
    scheduler = utils.LearningRateSchedulerLinear(optimizer, start_factor=1, end_factor=0.1, n_epochs=epochs)
    # scheduler = utils.LearningRateSchedulerStep(optimizer, step_size=5, gamma=0.95)

    # Instantiate early stopping
    early_stopping = utils.EarlyStopping(patience=epochs, verbose=False)

    # Initialize loss the plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], '-', label='loss train', color='red', linewidth=0.5)
    line2, = ax.plot([], [], '-', label='loss val', color='blue', linewidth=0.5)
    # line3, = ax.plot([], [], '-', label='loss test', color='green', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xlabel(r'epoch')
    ax.set_ylabel(r'loss')
    ax.set_title(r'Loss evolution')
    ax.legend()

    # Update model in the training process and test it
    start_time = time.time()
    df_training = pd.DataFrame(columns=('epoch', 'loss_train'))
    df_validation = pd.DataFrame(columns=('epoch', 'loss_val'))
    # df_test = pd.DataFrame(columns=('epoch', 'loss_test'))
    for t in range(epochs): # epochs is defined in the hyperparameters section above
        print(f'Epoch {t+1}')
        train(training_data, model, loss_function, optimizer, device, df_training, epoch=t)
        epoch_val_loss = val(validation_data, model, loss_function, device, df_validation, epoch=t)

        # learning rate scheduling
        scheduler(epoch_val_loss)

        # Early stopping
        early_stopping(epoch_val_loss, model, path='checkpoints')
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Update the plot
        line1.set_data(df_training['epoch'], df_training['loss_train'])
        line2.set_data(df_validation['epoch'], df_validation['loss_val'])
        # line3.set_data(df_test['epoch'], df_test['loss_test'])
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)  # Pause to update the plot

        print('-------------------------------')
    
    print("Done! ---Execution time: %s seconds ---" % (time.time() - start_time))

    # Save results
    utils.logger_cnn(run=run, batches=batch_size, input_channels=input_channels, channels=channels, d_fc=d_fc, lr=lr, epochs=epochs)

    # Finalize the plot
    plt.ioff()  # Turn off interactive mode

    # Save and close the plot
    plt.savefig(f'results/run_u_{run}/loss.png', dpi=300)
    plt.close()

    # Save the model
    torch.save(model, f'results/run_u_{run}/unet_model.pth')
    print(f'Saved PyTorch entire model to results/run_t_{run}/unet_model.pth')

    # # Load the model
    # model = torch.load(f'results/run_u_{run}/unet_model.pth').to(device)
    # print(f'Loaded PyTorch model from results/run_u_{run}/unet_model.pth')

    # Inference
    tgts_train, tgt_hats_train = test(training_data_inference, model, 'train', device)
    tgts_val, tgt_hats_val = test(validation_data_inference, model, 'validation', device)
    tgts_test, tgt_hats_test = test(testing_data, model, 'test', device)

    # Plot testing results
    utils.plots_unet(tgts_train, tgt_hats_train, station=station, phase='train', run=run)
    utils.plots_unet(tgts_val, tgt_hats_val, station=station, phase='validation', run=run)
    utils.plots_unet(tgts_test, tgt_hats_test, station=station, phase='test', run=run)

    # Metrics
    utils.metrics_unet(truth=tgts_train, hat=tgt_hats_train, phase='train', run=run)
    utils.metrics_unet(truth=tgts_val, hat=tgt_hats_val, phase='validation', run=run)
    utils.metrics_unet(truth=tgts_test, hat=tgt_hats_test, phase='test', run=run)