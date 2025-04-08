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

import utils
import dataset as ds
import models.transformer as tst

# Define the training step
def train(dataloader, model, src_mask, memory_mask, tgt_mask, loss_function, optimizer, device, df_training, epoch):
    
    num_batches = len(dataloader)
    training_loss = 0.0 # Initialize as a scalar for cumulative loss

    # Set model to training mode
    model.train()

    # Implementation without gradient scaling
    for i, batch in enumerate(dataloader):
        src, tgt, tgt_y, src_p, tgt_p = batch

        # Send data to device
        src, tgt, tgt_y, src_p, tgt_p = src.to(device), tgt.to(device), tgt_y.to(device), src_p.to(device), tgt_p.to(device)

        # Zero out gradients for every batch
        optimizer.zero_grad()
        
        # Forward pass
        y_hat, sa_weights_encoder, sa_weights, mha_weights = model(src=src, tgt=tgt, src_mask=src_mask, memory_mask=memory_mask, tgt_mask=tgt_mask)

        # Compute loss
        loss = loss_function(y_hat, tgt_y)
        
        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)

        # Update weights
        optimizer.step()

        # Accumulate loss for the epoch
        training_loss += loss.item()
    
    # Calculate the average training loss for the epoch
    epoch_train_loss = training_loss / num_batches

    # Append to the DataFrame
    df_training.loc[len(df_training)] = [epoch, epoch_train_loss]

# Define validation step
def val(dataloader, model, src_mask, memory_mask, tgt_mask, loss_function, device, df_validation, epoch):
    
    num_batches = len(dataloader)
    validation_loss = 0.0 # Initialize as a scalar for cumulative loss

    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            src, tgt, tgt_y, src_p, tgt_p = batch
            # Send data to device
            src, tgt, tgt_y, src_p, tgt_p = src.to(device), tgt.to(device), tgt_y.to(device), src_p.to(device), tgt_p.to(device)
            
            # Forward pass
            y_hat, sa_weights_encoder, sa_weights, mha_weights = model(src=src, tgt=tgt, src_mask=src_mask, memory_mask=memory_mask, tgt_mask=tgt_mask)
            
            # Compute loss
            loss = loss_function(y_hat, tgt_y)

            # Accumulate loss for the epoch
            validation_loss += loss.item()
        
        # Calculate the average validation loss for the epoch
        epoch_val_loss = validation_loss / num_batches

        # Append to the DataFrame
        df_validation.loc[len(df_validation)] = [epoch, epoch_val_loss]

    return epoch_val_loss

# Define inference step
def test(dataloader, model, src_mask, memory_mask, tgt_mask, phase, dates, device):
    
    # Get test dates
    # dates = data.index[testing_indices[0][0]:testing_indices[-1][1]]

    # Set up objects to store metrics for each instance
    src, tgt, tgt_y, src_p, tgt_p = next(iter(dataloader)) # Get the first batch to get the shape of the tensors
    tgt_ys = torch.zeros((len(dataloader), tgt.shape[1], tgt.shape[2]), device=device)  # Shape: [num_batches, seq_len, num_features]
    y_hats = torch.zeros((len(dataloader), tgt.shape[1], tgt.shape[2]), device=device)  # Shape: [num_batches, seq_len, num_features]

    # Define dictionary to store the attention weights, ground truth and predictions
    results = {}

    # Perform inference
    model.eval()
    with torch.no_grad():
        for i, (src, tgt, tgt_y, src_p, tgt_p) in enumerate(dataloader):

            # Send data to device
            src, tgt, tgt_y, src_p, tgt_p = src.to(device), tgt.to(device), tgt_y.to(device), src_p.to(device), tgt_p.to(device)

            # Forward pass
            y_hat, sa_weights_encoder, sa_weights, mha_weights = model(src=src, tgt=tgt, src_mask=src_mask, memory_mask=memory_mask, tgt_mask=tgt_mask)

            # Store ground truth and prediction
            tgt_ys[i] = tgt_y
            y_hats[i] = y_hat

            # Squeeze and pass src, tgt_y, pred, and weights to the cpu for plotting purposes
            src = src.squeeze(0).cpu().detach().numpy()
            tgt_y = tgt_y.squeeze(0).cpu().detach().numpy()
            y_hat = y_hat.squeeze(0).cpu().detach().numpy()
            sa_weights_encoder = sa_weights_encoder.squeeze(0).cpu().detach().numpy()

            # Save the results
            if phase == 'test':
                results[dates[i]] = {
                    'src': src,
                    'tgt_y': tgt_y,
                    'y_hat': y_hat,
                    'weights': sa_weights_encoder,
                }
    
    # Save the dictionary to numpy file
    if phase != '': # if phase == 'test':
        np.save(f'results/run_t_{run}/results_{phase}.npy', results, allow_pickle=True, fix_imports=True)
    
    # Pass tgt_ys and y_hats to CPU and convert to numpy arrays
    tgt_ys = tgt_ys.cpu().numpy()
    y_hats = y_hats.cpu().numpy()

    return tgt_ys, y_hats

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
    task_type = 'MM' # MU for multivariate predicts univariate, MM for multivariate predicts multivariate

    # Define run number
    run = 0
    
    # Hyperparams
    batch_size = 128
    validation_size = 0.125
    src_variables = [f'ammonium_{station}', f'conductivity_{station}', 
                    f'dissolved_oxygen_{station}', f'pH_{station}', 
                    f'precipitation_{station}', f'turbidity_{station}',
                    f'water_temperature_{station}', 'label']
    tgt_variables = [f'ammonium_{station}', f'conductivity_{station}', 
                    f'dissolved_oxygen_{station}', f'pH_{station}', 
                    f'precipitation_{station}', f'turbidity_{station}',
                    f'water_temperature_{station}', 'label']

    # input_variables would be just the src or tgt because we are predicting the tgt from the src, and not a tgt that is not in the src
    input_variables = src_variables
    timestamp_col_name = "date"

    # Only use data from this date and onwards
    cutoff_date = datetime.datetime(2005, 1, 1) 

    d_model = 16
    n_heads = 2
    n_encoder_layers = 1
    n_decoder_layers = 1 # Remember that with the current implementation it always has a decoder layer that returns the weights
    encoder_sequence_len = 384 # length of input given to encoder used to create the pre-summarized windows
    decoder_sequence_len = 96 # length of input given to decoder
    output_sequence_len = 96 # Target sequence length.
    in_features_encoder_linear_layer = 128
    in_features_decoder_linear_layer = 128
    max_sequence_len = encoder_sequence_len
    window_size = encoder_sequence_len + output_sequence_len # Used to slice data into sub-sequences
    step_size = 96 # Step size, i.e. how many time steps does the moving window move at each step
    batch_first = True

    # Run parameters
    lr = 0.0005
    epochs = 300

    # Get device
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using {device} device')

    # Read data
    data = utils.read_data(timestamp_col_name=timestamp_col_name)
    
    # Define the training and validation bounds
    training_val_lower_bound = datetime.datetime(2005, 1, 1)
    training_val_upper_bound = datetime.datetime(2017, 12, 31)

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
    dates = (pd.Series(data.index.date).drop_duplicates().sort_values()).to_list()  # Remove duplicates and sort
    print(dates)
    # Subset the test and validation sets
    dates_train_validation = dates[int(window_size/step_size):len(training_validation_indices)]
    dates_train = dates_train_validation[:(round(len(training_validation_indices) * (1-validation_size)))]
    dates_validation = dates_train_validation[-round(len(training_validation_indices) * validation_size):]
    
    # # Get the complete range of dates for the test set
    dates_test = dates[len(training_validation_indices):]
    
    # Make instance of the custom dataset class
    training_data = ds.TransformerDataset(task_type=task_type, data=torch.tensor(data[input_variables].values).float(),
                                        indices=training_indices, encoder_sequence_len=encoder_sequence_len, 
                                        decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)
    validation_data = ds.TransformerDataset(task_type=task_type, data=torch.tensor(data[input_variables].values).float(),
                                        indices=validation_indices, encoder_sequence_len=encoder_sequence_len, 
                                        decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)
    testing_data = ds.TransformerDataset(task_type=task_type, data=torch.tensor(data[input_variables].values).float(),
                                        indices=testing_indices, encoder_sequence_len=encoder_sequence_len, 
                                        decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)

    # Set up the train and validation dataloaders used for inference
    training_data_inference = DataLoader(training_data, batch_size=1, shuffle=False)
    validation_data_inference = DataLoader(validation_data, batch_size=1, shuffle=False)

    # Set up the dataloaders
    training_data = DataLoader(training_data, batch_size, shuffle=True)
    validation_data = DataLoader(validation_data, batch_size, shuffle=True)
    testing_data = DataLoader(testing_data, batch_size=1, shuffle=False)

    # Instantiate the transformer model and send it to device
    model = tst.TimeSeriesTransformer(input_size=len(src_variables), encoder_sequence_len=encoder_sequence_len, decoder_sequence_len=decoder_sequence_len,
                                    batch_first=batch_first, d_model=d_model, n_encoder_layers=n_encoder_layers, n_decoder_layers=n_decoder_layers, 
                                    n_heads=n_heads, dropout_encoder=0.2, dropout_decoder=0.2, dropout_pos_encoder=0.1, 
                                    dim_feedforward_encoder=in_features_encoder_linear_layer, dim_feedforward_decoder=in_features_decoder_linear_layer, 
                                    num_src_features=len(src_variables), num_predicted_features=len(tgt_variables)).to(device)
    
    # Print model and number of parameters
    print('Defined model:\n', model)
    utils.count_parameters(model)
    
    # Make src mask for the decoder with size
    # [batch_size*n_heads, output_sequence_length, encoder_sequence_len]
    src_mask = utils.masker(dim1=encoder_sequence_len, dim2=encoder_sequence_len).to(device)
    
    # Make the memory mask for the decoder
    memory_mask = None

    # Make tgt mask for decoder with size
    # [batch_size*n_heads, output_sequence_length, output_sequence_length]
    tgt_mask = utils.masker(dim1=output_sequence_len, dim2=output_sequence_len).to(device)
    
    # Define optimizer and loss function
    loss_function = nn.MSELoss()
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
        train(training_data, model, src_mask, memory_mask, tgt_mask, loss_function, optimizer, device, df_training, epoch=t)
        epoch_val_loss = val(validation_data, model, src_mask, memory_mask, tgt_mask, loss_function, device, df_validation, epoch=t)

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
    utils.logger_transformer(run=run, batches=batch_size, d_model=d_model, n_heads=n_heads,
                encoder_layers=n_encoder_layers, decoder_layers=n_decoder_layers,
                dim_ll_encoder=in_features_encoder_linear_layer, dim_ll_decoder=in_features_decoder_linear_layer,
                lr=lr, loss=epoch_val_loss, epochs=epochs, seed=seed, run_time=(time.time() - start_time))
    
    # Finalize the plot
    plt.ioff()  # Turn off interactive mode

    # Save and close the plot
    plt.savefig(f'results/run_t_{run}/loss.png', dpi=300)
    plt.close()

    # Save the model
    torch.save(model, f'results/run_t_{run}/transformer_model.pth')
    print(f'Saved PyTorch entire model to results/run_t_{run}/model.pth')

    # # Load the model
    # model = torch.load("results/run_t_{run}/transformer_model.pth", map_location=torch.device('cpu')).to(device)
    # print('Loaded PyTorch model from results/models/transformer_model.pth')

    # Inference
    # tgt_ys_train, y_hats_train = test(training_data, model, src_mask, memory_mask, tgt_mask, 'train', dates_train, device)
    # tgt_ys_val, y_hats_val = test(validation_data, model, src_mask, memory_mask, tgt_mask, 'validation', dates_validation, device)
    tgt_ys_test, y_hats_test = test(testing_data, model, src_mask, memory_mask, tgt_mask, 'test', dates_test, device)

    # Load results object
    phase = 'test'
    results = np.load(f'results/run_t_{run}/results_{phase}.npy', allow_pickle=True, fix_imports=True).item()  # Convert back to dict

    # Plot results for the first 25 dates
    plot_dates = [datetime.date(2020, 11, 22), datetime.date(2020, 11, 23), datetime.date(2020, 11, 24), datetime.date(2020, 11, 25), datetime.date(2020, 11, 26), datetime.date(2020, 11, 27),
                datetime.date(2022, 11, 16), datetime.date(2022, 11, 17), datetime.date(2022, 11, 18), datetime.date(2022, 11, 19), datetime.date(2022, 11, 20), datetime.date(2022, 11, 21)]
    for i, (date, data) in enumerate(results.items()):
        if date in plot_dates:
            utils.plots_transformer(date=date, src=data['src'], truth=data['tgt_y'], hat=data['y_hat'], weights=data['weights'], 
                                    tgt_percentage=1, station=station, phase=phase, 
                                    instance=date)

    # # Metrics
    # utils.metrics_transformer(tgt_ys_train, y_hats_train, 'train', run=run)
    # utils.metrics_transformer(tgt_ys_val, y_hats_val, 'validation', run=run)
    # utils.metrics_transformer(tgt_ys_test, y_hats_test, 'test', run=run)