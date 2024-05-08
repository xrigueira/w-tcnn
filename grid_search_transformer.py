import time
import datetime
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
import dataset as ds
import models.transformer as tst
import transformer as mn

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""Hyperparameter tuning with Grid Search."""

if __name__ == "__main__":

    # Define seed
    torch.manual_seed(0)

    # Define fixed hyperparameters
    validation_size = 0.125
    # src_variables = ['x1']
    src_variables = ['x1', 'x2']
    tgt_variables = ['y']
    input_variables = src_variables + tgt_variables
    timestamp_col_name = "time"

    # Get device
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using {device} device')

    # Define hyperparameters for grid search
    parameters = {"batch_size": [128, 256, 512], 
                "d_model": [32, 64, 128, 256, 512], 
                "n_heads": [2, 4, 8],
                "in_features_encoder_linear_layer": [256], 
                "in_features_decoder_linear_layer": [256],
                "lr": [0.00001, 0.0001, 0.001], 
                "epochs": [100, 150, 200, 300, 400, 800]
                }

    # Perform grid search
    results = pd.DataFrame(columns=['batch_size', 'd_model', 'n_heads', 'in_features_encoder_linear_layer', 'in_features_decoder_linear_layer', 'lr', 'epochs', 'NSE'])

    # Define number of trials
    trial = 0

    for batch_size in parameters["batch_size"]:
        for d_model in parameters["d_model"]:
            for n_heads in parameters["n_heads"]:
                for in_features_encoder_linear_layer in parameters["in_features_encoder_linear_layer"]:
                    for in_features_decoder_linear_layer in parameters["in_features_decoder_linear_layer"]:
                        for lr in parameters["lr"]:
                            for epochs in parameters["epochs"]:

                                # Only use data from this date and onwards
                                cutoff_date = datetime.datetime(1980, 1, 1)

                                n_encoder_layers = 1
                                n_decoder_layers = 1 # Remember that with the current implementation it always has a decoder layer that returns the weights
                                encoder_sequence_len = 1461 # length of input given to encoder used to create the pre-summarized windows (4 years of data) 1461
                                crushed_encoder_sequence_len = 53 # Encoder sequence length afther summarizing the data when defining the dataset 53
                                decoder_sequence_len = 1 # length of input given to decoder
                                output_sequence_len = 1 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
                                window_size = encoder_sequence_len + output_sequence_len # used to slice data into sub-sequences
                                step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
                                batch_first = True

                                # Read data
                                data = utils.read_data(timestamp_col_name=timestamp_col_name)
                                
                                # Extract train and val data for nomralization purposes
                                training_val_lower_bound = datetime.datetime(1980, 10, 1)
                                training_val_upper_bound = datetime.datetime(2007, 9, 30)

                                # Extract train/validation data
                                training_val_data = data[(training_val_lower_bound <= data.index) & (data.index <= training_val_upper_bound)]
                                
                                # Calculate the percentage of data in the training_val_data subset
                                total_data_range = (data.index.max() - data.index.min()).days
                                training_val_range = (training_val_upper_bound - training_val_lower_bound).days
                                train_val_percentage = (training_val_range / total_data_range)
                                
                                # Normalize the data
                                from sklearn.preprocessing import MinMaxScaler
                                scaler = MinMaxScaler()

                                # Fit scaler on the training set
                                scaler.fit(training_val_data.iloc[:, 1:])

                                # Transform the training and test sets
                                data.iloc[:, 1:] = scaler.transform(data.iloc[:, 1:])

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
                                training_data = ds.TransformerDataset(data=torch.tensor(data[input_variables].values).float(),
                                                                    indices=training_indices, encoder_sequence_len=encoder_sequence_len, 
                                                                    decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)
                                validation_data = ds.TransformerDataset(data=torch.tensor(data[input_variables].values).float(),
                                                                    indices=validation_indices, encoder_sequence_len=encoder_sequence_len, 
                                                                    decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)
                                testing_data = ds.TransformerDataset(data=torch.tensor(data[input_variables].values).float(),
                                                                    indices=testing_indices, encoder_sequence_len=encoder_sequence_len, 
                                                                    decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)

                                # Set up dataloaders
                                training_val_data = training_data + validation_data # For testing puporses
                                training_data = DataLoader(training_data, batch_size, shuffle=False)
                                validation_data = DataLoader(validation_data, batch_size, shuffle=False)
                                testing_data = DataLoader(testing_data, batch_size=1)
                                training_val_data = DataLoader(training_val_data, batch_size=1) # For testing puporses

                                # Update the encoder sequence length to its crushed version
                                encoder_sequence_len = crushed_encoder_sequence_len

                                # Instantiate the transformer model and send it to device
                                model = tst.TimeSeriesTransformer(input_size=len(src_variables), decoder_sequence_len=decoder_sequence_len, 
                                                                batch_first=batch_first, d_model=d_model, n_encoder_layers=n_encoder_layers, 
                                                                n_decoder_layers=n_decoder_layers, n_heads=n_heads, dropout_encoder=0.2, 
                                                                dropout_decoder=0.2, dropout_pos_encoder=0.1, dim_feedforward_encoder=in_features_encoder_linear_layer, 
                                                                dim_feedforward_decoder=in_features_decoder_linear_layer, num_src_features=len(src_variables), 
                                                                num_predicted_features=len(tgt_variables)).to(device)

                                # Send model to device
                                model.to(device)

                                # Make src mask for the decoder with size
                                # [batch_size*n_heads, output_sequence_length, encoder_sequence_len]
                                src_mask = utils.masker(dim1=encoder_sequence_len, dim2=encoder_sequence_len).to(device)
                                # src_mask = utils.generate_square_subsequent_mask(size=encoder_sequence_len).to(device)
                                
                                # Make the memory mask for the decoder
                                # memory_mask = utils.unmasker(dim1=output_sequence_len, dim2=encoder_sequence_len).to(device)
                                memory_mask = None

                                # Make tgt mask for decoder with size
                                # [batch_size*n_heads, output_sequence_length, output_sequence_length]
                                tgt_mask = utils.masker(dim1=output_sequence_len, dim2=output_sequence_len).to(device)
                                # tgt_mask = utils.generate_square_subsequent_mask(size=decoder_sequence_len).to(device)
                                
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
                                    mn.train(training_data, model, src_mask, memory_mask, tgt_mask, loss_function, optimizer, device, df_training, epoch=t)
                                    epoch_val_loss = mn.val(validation_data, model, src_mask, memory_mask, tgt_mask, loss_function, device, df_validation, epoch=t)

                                    # Early stopping
                                    early_stopping(epoch_val_loss, model, path='checkpoints')
                                    if early_stopping.early_stop:
                                        print("Early stopping")
                                        break
                                
                                print("Done! ---Execution time: %s seconds ---" % (time.time() - start_time))
                                print(f"Running trial {trial} with parameters: batch_size {batch_size}, d_model {d_model}, n_heads {n_heads}, nn_enc {in_features_encoder_linear_layer}, nn_dec {in_features_decoder_linear_layer}, lr {lr}, epochs {epochs}.")
                                
                                # Get NSE metric on inference
                                tgt_y_truth_test, tgt_y_hat_test = mn.test(testing_data, model, src_mask, memory_mask, tgt_mask, device)

                                nse = utils.get_nash_sutcliffe_efficiency(tgt_y_truth_test, tgt_y_hat_test)
                                print(f"NSE: {nse}")

                                # Update trial
                                trial += 1

                                # Update the results
                                results.loc[len(results.index)] = [batch_size, d_model, n_heads, in_features_encoder_linear_layer, in_features_decoder_linear_layer, lr, epochs, nse]

                                # Save the results at each step
                                results.to_csv(f'results/results.csv', sep=',', encoding='utf-8', index=True)