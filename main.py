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
import transformer as tst

# Define the training step
def train(dataloader, model, src_mask, memory_mask, tgt_mask, loss_function, optimizer, device, df_training, epoch):
    
    size = len(dataloader.dataset)
    model.train()
    training_loss = [] # For plotting purposes
    for i, batch in enumerate(dataloader):
        src, tgt, tgt_y, src_p, tgt_p = batch
        src, tgt, tgt_y, src_p, tgt_p = src.to(device), tgt.to(device), tgt_y.to(device), src_p.to(device), tgt_p.to(device)

        # Zero out gradients for every batch
        optimizer.zero_grad()
        
        # Compute prediction error
        pred, sa_weights_encoder, sa_weights, mha_weights = model(src=src, tgt=tgt, src_mask=src_mask, memory_mask=memory_mask, tgt_mask=tgt_mask)
        pred = pred.to(device)
        loss = loss_function(pred, tgt_y)
        
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

# Define testing step
def val(dataloader, model, src_mask, memory_mask, tgt_mask, loss_function, device, df_validation, epoch):
    
    num_batches = len(dataloader)
    model.eval()
    validation_loss = [] # For plotting purposes
    with torch.no_grad():
        for batch in dataloader:
            src, tgt, tgt_y, src_p, tgt_p = batch
            src, tgt, tgt_y, src_p, tgt_p = src.to(device), tgt.to(device), tgt_y.to(device), src_p.to(device), tgt_p.to(device)
            
            pred, sa_weights_encoder, sa_weights, mha_weights = model(src=src, tgt=tgt, src_mask=src_mask, memory_mask=memory_mask, tgt_mask=tgt_mask)
            pred = pred.to(device)
            loss = loss_function(pred, tgt_y)
            
            # Save results for plotting
            validation_loss.append(loss.item())
            epoch_val_loss = np.mean(validation_loss)
            df_validation.loc[epoch] = [epoch, epoch_val_loss]
    
    loss /= num_batches
    # print(f"Avg test loss: {loss:>8f}")
    return epoch_val_loss

# Define inference step
def test(dataloader, model, src_mask, memory_mask, tgt_mask, device):
    
    # Get tgt for plotting purposes
    tgt_plots = torch.zeros(len(dataloader), output_sequence_len, len(tgt_variables))
    for i, (src, tgt, tgt_y, src_p, tgt_p) in enumerate(dataloader):
        tgt_plots[i] = tgt
    
    # Get ground truth
    tgt_y_truth = torch.zeros(len(dataloader), output_sequence_len, len(tgt_variables))
    for i, (src, tgt, tgt_y, src_p, tgt_p) in enumerate(dataloader):
        tgt_y_truth[i] = tgt_y

    # Define tensor to store the predictions
    tgt_y_hat = torch.zeros((len(dataloader)), output_sequence_len, len(tgt_variables), device=device)

    # Define list to store the attention weights
    all_sa_weights_encoder_inference = []
    all_sa_weights_inference = []
    all_mha_weights_inference = []
    
    # Perform inference
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            src, tgt, tgt_y, src_p, tgt_p = sample
            src, tgt, tgt_y, src_p, tgt_p = src.to(device), tgt.to(device), tgt_y.to(device), src_p.to(device), tgt_p.to(device)

            pred, sa_weights_encoder, sa_weights, mha_weights = model(src=src, tgt=tgt, src_mask=src_mask, memory_mask=memory_mask, tgt_mask=tgt_mask)
            # all_sa_weights_encoder_inference.append(sa_weights_encoder)
            # all_sa_weights_inference.append(sa_weights)
            # all_mha_weights_inference.append(mha_weights)
            pred = pred.to(device)

            # # Save src, tgt and tgt_y, and pred for plotting purposes
            # np.save(f'results/src_p_{i}.npy', src_p.cpu(), allow_pickle=False, fix_imports=False)
            # np.save(f'results/tgt_p_{i}.npy', tgt_p.cpu(), allow_pickle=False, fix_imports=False)
            # np.save(f'results/tgt_y_hat_{i}.npy', pred.cpu(), allow_pickle=False, fix_imports=False)
            
            tgt_y_hat[i] = pred
    
    # # Save inference attention for the last step
    # np.save('results/all_sa_encoder_weights.npy', np.stack([sa_weight_encoder.cpu().numpy() for sa_weight_encoder in all_sa_weights_encoder_inference]), allow_pickle=False, fix_imports=False)
    # np.save('results/all_sa_weights.npy', np.stack([sa_weight.cpu().numpy() for sa_weight in all_sa_weights_inference]), allow_pickle=False, fix_imports=False)
    # np.save('results/all_mha_weights.npy', np.stack([mha_weight.cpu().numpy() for mha_weight in all_mha_weights_inference]), allow_pickle=False, fix_imports=False)
    
    # Pass target_y_hat to cpu for plotting purposes
    tgt_y_hat = tgt_y_hat.cpu()

    tgt_y_truth, tgt_y_hat = tgt_y_truth.numpy(), tgt_y_hat.numpy()

    # Save tgt_plots, ground truth and predictions
    np.save('tgt_plots.npy', tgt_plots, allow_pickle=False, fix_imports=False)
    np.save('tgt_y_truth.npy', tgt_y_truth, allow_pickle=False, fix_imports=False)
    np.save('tgt_y_hat.npy', tgt_y_hat, allow_pickle=False, fix_imports=False)
    
    return tgt_plots, tgt_y_truth, tgt_y_hat

if __name__ == '__main__':
    
    # Define seed
    torch.manual_seed(0)

    station = 901
    task_type = 'MM' # MU for multivariate predicts univariate, MM for multivariate predicts multivariate

    # Define run number
    run = 1
    
    # Hyperparams
    batch_size = 16
    validation_size = 0.125
    src_variables = [f'ammonium_{station}', f'conductivity_{station}', 
                    f'dissolved_oxygen_{station}', f'pH_{station}', 
                    f'precipitation_{station}', f'turbidity_{station}',
                    f'water_temperature_{station}']
    tgt_variables = [f'ammonium_{station}', f'conductivity_{station}', 
                    f'dissolved_oxygen_{station}', f'pH_{station}', 
                    f'precipitation_{station}', f'turbidity_{station}',
                    f'water_temperature_{station}']
    # input_variables would be just the src or tgt because we are predicting the tgt from the src, and not a tgt that is not in the src
    input_variables = src_variables
    timestamp_col_name = "date"

    # Only use data from this date and onwards
    cutoff_date = datetime.datetime(2005, 1, 1) 

    d_model = 16
    n_heads = 2
    n_encoder_layers = 1
    n_decoder_layers = 1 # Remember that with the current implementation it always has a decoder layer that returns the weights
    encoder_sequence_len = 2016 # length of input given to encoder used to create the pre-summarized windows (4 years of data) 1461
    # crushed_encoder_sequence_len = 53 # Encoder sequence length afther summarizing the data when defining the dataset 53
    decoder_sequence_len = 672 # length of input given to decoder
    output_sequence_len = 672 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
    in_features_encoder_linear_layer = 256
    in_features_decoder_linear_layer = 256
    max_sequence_len = encoder_sequence_len
    window_size = encoder_sequence_len + output_sequence_len # used to slice data into sub-sequences
    step_size = 96 # Step size, i.e. how many time steps does the moving window move at each step
    batch_first = True

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
    training_val_data = data[(training_val_lower_bound <= data.index) & (data.index <= training_val_upper_bound)]
    
    # Calculate the percentage of data in the training_val_data subset
    total_data_range = (data.index.max() - data.index.min()).days
    training_val_range = (training_val_upper_bound - training_val_lower_bound).days
    train_val_percentage = (training_val_range / total_data_range)
    
    # # Normalize the data [already done in the smoother.py file]
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()

    # # Fit scaler on the training set
    # scaler.fit(training_val_data.iloc[:, 1:-1])

    # # Transform the training and test sets
    # data.iloc[:, 1:-1] = scaler.transform(data.iloc[:, 1:-1])
    
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
    training_data = ds.TransformerDataset(task_type=task_type, data=torch.tensor(data[input_variables].values).float(),
                                        indices=training_indices, encoder_sequence_len=encoder_sequence_len, 
                                        decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)
    validation_data = ds.TransformerDataset(task_type=task_type, data=torch.tensor(data[input_variables].values).float(),
                                        indices=validation_indices, encoder_sequence_len=encoder_sequence_len, 
                                        decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)
    testing_data = ds.TransformerDataset(task_type=task_type, data=torch.tensor(data[input_variables].values).float(),
                                        indices=testing_indices, encoder_sequence_len=encoder_sequence_len, 
                                        decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)

    # Set up dataloaders
    training_val_data = training_data + validation_data # For testing puporses
    training_data = DataLoader(training_data, batch_size, shuffle=False)
    validation_data = DataLoader(validation_data, batch_size, shuffle=False)
    testing_data = DataLoader(testing_data, batch_size=1, shuffle=False)
    training_val_data = DataLoader(training_val_data, batch_size=1, shuffle=False) # For testing puporses
    
    # # Update the encoder sequence length to its crushed version
    # encoder_sequence_len = crushed_encoder_sequence_len

    # Instantiate the transformer model and send it to device
    model = tst.TimeSeriesTransformer(input_size=len(src_variables), decoder_sequence_len=decoder_sequence_len, 
                                    batch_first=batch_first, d_model=d_model, n_encoder_layers=n_encoder_layers, 
                                    n_decoder_layers=n_decoder_layers, n_heads=n_heads, dropout_encoder=0.2, 
                                    dropout_decoder=0.2, dropout_pos_encoder=0.1, dim_feedforward_encoder=in_features_encoder_linear_layer, 
                                    dim_feedforward_decoder=in_features_decoder_linear_layer, num_src_features=len(src_variables), 
                                    num_predicted_features=len(tgt_variables)).to(device)

    # Send model to device
    model.to(device)
    
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

    # Instantiate early stopping
    early_stopping = utils.EarlyStopping(patience=30, verbose=True)

    # Update model in the training process and test it
    start_time = time.time()
    df_training = pd.DataFrame(columns=('epoch', 'loss_train'))
    df_validation = pd.DataFrame(columns=('epoch', 'loss_val'))
    for t in range(epochs): # epochs is defined in the hyperparameters section above
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_data, model, src_mask, memory_mask, tgt_mask, loss_function, optimizer, device, df_training, epoch=t)
        epoch_val_loss = val(validation_data, model, src_mask, memory_mask, tgt_mask, loss_function, device, df_validation, epoch=t)

        # Early stopping
        early_stopping(epoch_val_loss, model, path='checkpoints')
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    print("Done! ---Execution time: %s seconds ---" % (time.time() - start_time))

    # # Save the model
    # torch.save(model, "models/model.pth")
    # print("Saved PyTorch entire model to models/model.pth")

    # # Load the model
    # model = torch.load("models/model.pth").to(device)
    # print('Loaded PyTorch model from models/model.pth')

    # Inference
    tgt_plots, tgt_y_truth_train_val, tgt_y_hat_train_val = test(training_val_data, model, src_mask, memory_mask, tgt_mask, device)
    tgt_plots, tgt_y_truth_test, tgt_y_hat_test = test(testing_data, model, src_mask, memory_mask, tgt_mask, device)
    
    # # Save results
    # utils.logger(run=run, batches=batch_size, d_model=d_model, n_heads=n_heads,
    #             encoder_layers=n_encoder_layers, decoder_layers=n_decoder_layers,
    #             dim_ll_encoder=in_features_encoder_linear_layer, dim_ll_decoder=in_features_decoder_linear_layer,
    #             lr=lr, epochs=epochs)

    # # Plot loss
    # plt.figure(1);plt.clf()
    # plt.plot(df_training['epoch'], df_training['loss_train'], '-o', label='loss train')
    # plt.plot(df_training['epoch'], df_validation['loss_val'], '-o', label='loss val')
    # plt.yscale('log')
    # plt.xlabel(r'epoch')
    # plt.ylabel(r'loss')
    # plt.legend()
    # # plt.show()

    # plt.savefig(f'results/run_{run}/loss.png', dpi=300)

    # # Plot testing results
    # utils.plots(tgt_y_truth_train_val, tgt_y_hat_train_val, 'train_val', run=run)
    # utils.plots(tgt_y_truth_test, tgt_y_hat_test, 'test', run=run)

    # # Metrics
    # utils.metrics(tgt_y_truth_train_val, tgt_y_hat_train_val, 'train_val')
    # utils.metrics(tgt_y_truth_test, tgt_y_hat_test, 'test')
