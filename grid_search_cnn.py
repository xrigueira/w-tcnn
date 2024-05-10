import time
import datetime
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
import dataset as ds
import models.cnn as cnn
import unet

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""Hyperparameter tuning with Grid Search."""

if __name__ == "__main__":

    # Define seed
    torch.manual_seed(0)

    station = 901

    # Define fixed hyperparameters
    validation_size = 0.125
    variables = [f'ammonium_{station}', f'conductivity_{station}', 
                f'dissolved_oxygen_{station}', f'pH_{station}', 
                f'precipitation_{station}', f'turbidity_{station}',
                f'water_temperature_{station}', 'label']
    input_variables = variables
    timestamp_col_name = "date"

    # Get device
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using {device} device')

    # Define hyperparameters for grid search
    parameters = {"batch_size": [128, 256, 512], 
                "channels": [32, 64, 128, 256, 512], 
                "d_fc": [2, 4, 8],
                "lr": [0.00001, 0.0001, 0.001], 
                "epochs": [100, 150, 200, 300, 400, 800]
                }

    # Perform grid search
    results = pd.DataFrame(columns=['batch_size', 'd_model', 'n_heads', 'in_features_encoder_linear_layer', 'in_features_decoder_linear_layer', 'lr', 'epochs', 'NSE'])

    # Define number of trials
    trial = 0

    for batch_size in parameters["batch_size"]:
        for channels in parameters["channels"]:
            for d_fc in parameters["d_fc"]:
                for lr in parameters["lr"]:
                    for epochs in parameters["epochs"]:

                        # Only use data from this date and onwards
                        cutoff_date = datetime.datetime(2005, 1, 1)

                        n_variables = len(variables) - 1 # Exclude the label
                        window_size = 96 # Used to slice data into sub-sequences
                        step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step

                        input_channels = 1
                        n_classes = 1 # Anomaly or non anomaly

                        # Read data
                        data = utils.read_data(timestamp_col_name=timestamp_col_name)
                        
                        # Extract train and val data for nomralization purposes
                        training_val_lower_bound = datetime.datetime(2005, 1, 1)
                        training_val_upper_bound = datetime.datetime(2017, 12, 31)

                        # Extract train/validation data
                        training_val_data = data[(training_val_lower_bound <= data.index) & (data.index <= training_val_upper_bound)]
                        
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

                        # Instantiate the cnn model and send it to device
                        n_variables, window_size, n_classes, input_channels, channels, d_fc
                        model = cnn.UNet(n_variables=n_variables, window_size=window_size, n_classes=n_classes,
                                        input_channels=input_channels, channels=channels, d_fc=d_fc).to(device)

                        # Send model to device
                        model.to(device)
                        
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
                            # print(f"Epoch {t+1}\n-------------------------------")
                            unet.train(training_data, model, loss_function, optimizer, device, df_training, epoch=t)
                            epoch_val_loss = unet.val(validation_data, model, loss_function, device, df_validation, epoch=t)

                            # Early stopping
                            early_stopping(epoch_val_loss, model, path='checkpoints')
                            if early_stopping.early_stop:
                                print("Early stopping")
                                break
                        
                        print("Done! ---Execution time: %s seconds ---" % (time.time() - start_time))
                        print(f"Running trial {trial} with parameters: batch_size {batch_size}, channels {channels}, d_fc {d_fc}, lr {lr}, epochs {epochs}.")
                        
                        # Get NSE metric on inference
                        tgt_truth_test, tgt_hat_test = unet.test(testing_data, model, 'test', device)

                        rmse, r2 = utils.metrics_cnn(tgt_truth_test, tgt_hat_test, 'test')
                        print(f"NSE: {rmse}")

                        # Update trial
                        trial += 1

                        # Update the results
                        results.loc[len(results.index)] = [batch_size, channels, d_fc, lr, epochs, rmse]

                        # Save the results at each step
                        results.to_csv(f'results/results_cnn.csv', sep=',', encoding='utf-8', index=True)