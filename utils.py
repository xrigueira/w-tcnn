import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.stats import pearsonr
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch import Tensor

from pathlib import Path
from typing import Optional, Any, Union, Callable, Tuple

def masker(dim1: int, dim2: int) -> Tensor:
    
    """
    Generates an upper-triangular matrix of -inf, with
    zeros on the diagonal. Modified from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    ----------
    Arguments:
    dim1: int, for both src and tgt masking, this must be
        a sequence length
    dim2: int, for src masking this must be encoder sequence
        length (i.e. the length of the input sequence to the model),
        and for tgt masking , this must be target sequence length
    
    Return:
    A tensor of shape [dim1, dim2]
    """
    
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

def get_indices(data: pd.DataFrame, window_size: int, step_size: int) -> list:
    
    """
    Produce all the start and end index position that is needed to obtain the sub-sequences.
    
    Returns a list of tuples. Each tuple is (start_idx, end_idx) of a subsequence. These tuples
    should be used to slice the dataset into sub-sequences. These sub-sequences should then be
    passed into a function that sliced them into input and target sequences.
    ----------
    Arguments:
    data (pd.DataFrame): loaded database to generate the subsequences from.
    window_size (int): the desired length of each sub-sequence. Should be (input_sequence_length + 
        tgt_sequence_length). E.g. if you want the model to consider the past 100 time steps in 
        order to predict the future 50 time_steps, window_size = 100 + 50 = 150.
    step_size (int): size of each step as the data sequence is traversed by the moving window.
    
    Return:
    indices: a lits of tuples.
    """
    
    # Define the stop position
    stop_position = len(data) - 1 # because of 0 indexing in Python
    
    # Start the first sub-sequence at index 0
    subseq_first_idx = 0
    subseq_last_idx = window_size
    
    indices = []
    while subseq_last_idx <= stop_position:
        
        indices.append((subseq_first_idx, subseq_last_idx))
        
        subseq_first_idx += step_size
        subseq_last_idx += step_size
        
    return indices

def read_data(data_dir: Union[str, Path] = 'data', station: int=901, timestamp_col_name: str='date') -> pd.DataFrame:
    
    """Read data from csv file and return a pd.DataFrame object.
    ----------
    Arguments:
    data_dir: str or Path object specifying the path to the directory containing the data.
    station: int, the station number to read data from.
    tgt_col_name: str, the name of the column containing the target variable.
    timestamp_col_name: str, the name of the column or named index containing the timestamps.
    
    Returns:
    data (pd.DataFrame): data read an loaded as a Pandas DataFrame
    """
    
    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)
    
    # Read smothed csv file
    csv_files = list(data_dir.glob(f"*{station}_smo.csv"))
    
    if len(csv_files) > 1:
        raise ValueError("data_dir contains more than 1 csv file. Must only contain 1")
    elif len(csv_files) == 0:
        raise ValueError("data_dir must contain at least 1 csv file.")

    data_path = csv_files[0]

    print("Reading file in {}".format(data_path))
    
    data = pd.read_csv(data_path, parse_dates=[timestamp_col_name], index_col=[timestamp_col_name],  low_memory=False)
    
    # Make sure all "n/e" values have been removed from df. 
    if ne_check(data):
        raise ValueError("data frame contains 'n/e' values. These must be handled")

    # Downcast columns to smallest possible version
    data = to_numeric_and_downcast_data(data)

    # Make sure data is in ascending order by timestamp
    data.sort_values(by=[timestamp_col_name], inplace=True)

    return data

def ne_check(df:pd.DataFrame):
    
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """
    
    for col in df.columns:

        true_bool = (df[col] == "n/e")

        if any(true_bool):
            return True

    return False

def to_numeric_and_downcast_data(df: pd.DataFrame):
    
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    
    fcols = df.select_dtypes('float').columns
    
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df

# Define function to get and format the number of parameters
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    
    print(table)
    print(f"Total trainable parameters: {total_params}")
    
    return total_params

# Define a class for early stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

class LearningRateSchedulerPlateau:
    
    def __init__(self, optimizer, patience, factor):
        self.optimizer = optimizer
        self.mode = 'min'
        self.factor = factor
        self.patience = patience
        self.threshold = 1e-4
        self.threshold_mode = 'rel'
        self.cooldown = 0
        self.min_lr = 1e-6
        self.eps = 1e-8

        self.val_loss_min = np.Inf
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, self.mode, self.factor, self.patience,
                                                                    self.threshold, self.threshold_mode, self.cooldown, self.min_lr, 
                                                                    self.eps)
        self.prev_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]

    def __call__(self, val_loss):
        self.scheduler.step(val_loss)

        # Output the new learning rate if it is updated
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = param_group['lr']
            if new_lr != self.prev_lr[i]:
                print(f'Learning rate updated ({self.prev_lr[i]:.6f} --> {new_lr:.6f})')
                self.prev_lr[i] = new_lr
        
        # Output when the validation loss is reduced
        if val_loss < self.val_loss_min:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            self.val_loss_min = val_loss

class LearningRateSchedulerLinear:
    
    def __init__(self, optimizer, start_factor, end_factor, n_epochs):
        self.optimizer = optimizer
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.n_epochs = n_epochs
        
        self.val_loss_min = np.Inf
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=self.start_factor, 
                                                        end_factor=self.end_factor, total_iters=self.n_epochs, 
                                                        last_epoch=-1)
        self.prev_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]

    def __call__(self, val_loss):
        self.scheduler.step()

        # Output the new learning rate if it is updated
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = param_group['lr']
            if new_lr != self.prev_lr[i]:
                print(f'Learning rate updated ({self.prev_lr[i]:.6f} --> {new_lr:.6f})')
                self.prev_lr[i] = new_lr
        
        # Output when the validation loss is reduced
        if val_loss < self.val_loss_min:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            self.val_loss_min = val_loss

class LearningRateSchedulerStep:

    def __init__(self, optimizer, step_size, gamma):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma

        self.val_loss_min = np.Inf
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma,
                                                        last_epoch=-1)
        self.prev_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]

    def __call__(self, val_loss):
        self.scheduler.step()

        # Output the new learning rate if it is updated
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = param_group['lr']
            if new_lr != self.prev_lr[i]:
                print(f'Learning rate updated ({self.prev_lr[i]:.6f} --> {new_lr:.6f})')
                self.prev_lr[i] = new_lr
        
        # Output when the validation loss is reduced
        if val_loss < self.val_loss_min:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            self.val_loss_min = val_loss

# Define Nash-Sutcliffe efficiency
def get_nash_sutcliffe_efficiency(observed, modeled):
    # Add small random noise to avoid division by zero when observed is constant
    # observed = observed + np.random.normal(1e-4, 1e-6, size=observed.shape)
    # modeled = modeled + np.random.normal(1e-4, 1e-6, size=modeled.shape)

    mean_observed = np.mean(observed)
    numerator = np.sum((observed - modeled)**2)
    denominator = np.sum((observed - mean_observed)**2)
    
    # Calculate Nash-Sutcliffe efficiency
    nse = 1 - (numerator / denominator)
    
    return nse

def get_multivariate_nash_sutcliffe_efficiency(multivariate_observed, multivariate_modeled):

    multivariate_nse = []
    for i in range(multivariate_observed.shape[1]):
        nse = get_nash_sutcliffe_efficiency(multivariate_observed[:, i].flatten(), 
                                            multivariate_modeled[:, i].flatten())
        multivariate_nse.append(nse)
    
    return multivariate_nse

# Define function to calculate the percent bias
def get_pbias(observed, modeled):
    return np.sum(observed - modeled) / np.sum(observed) * 100

def get_multivariate_pbias(multivariate_observed, multivariate_modeled):

    multivariate_pbias = []
    for i in range(multivariate_observed.shape[1]):
        pbias = get_pbias(multivariate_observed[:, i].flatten(), 
                        multivariate_modeled[:, i].flatten())
        multivariate_pbias.append(pbias)
    
    return multivariate_pbias

# Define function to calculate the Kling-Gupta efficiency
def get_kge(observed, modeled):
    # Add small random noise to avoid division by zero when observed is constant
    # observed = observed + np.random.normal(0, 1e-10, size=observed.shape)
    # modeled = modeled + np.random.normal(0, 1e-10, size=modeled.shape)
    
    # Calculate the Kling-Gupta efficiency
    r = pearsonr(observed, modeled)[0]
    alpha = np.std(modeled) / np.std(observed)
    beta = np.sum(modeled) / np.sum(observed)

    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def get_multivariate_kge(multivariate_modeled, multivariate_observed):

    multivariate_kge = []
    for i in range(multivariate_observed.shape[1]):
        kge = get_kge(multivariate_observed[:, i].flatten(), 
                    multivariate_modeled[:, i].flatten())
        multivariate_kge.append(kge)
    
    return multivariate_kge

def get_multivariate_mse(multivariate_modeled, multivariate_observed):

    multivariate_mse = []
    for i in range(multivariate_observed.shape[1]):
        mse = mean_squared_error(multivariate_observed[:, i].flatten(), 
                                        multivariate_modeled[:, i].flatten())
        multivariate_mse.append(mse)
    
    return multivariate_mse

def get_multivariate_mae(multivariate_modeled, multivariate_observed):
    
        multivariate_mae = []
        for i in range(multivariate_observed.shape[1]):
            mae = np.mean(np.abs(multivariate_observed[:, i].flatten() - 
                                multivariate_modeled[:, i].flatten()))
            multivariate_mae.append(mae)
        
        return multivariate_mae

def get_multivariate_r2(multivariate_modeled, multivariate_observed):
    
    multivariate_r2 = []
    for i in range(multivariate_observed.shape[1]):
        r2 = r2_score(multivariate_observed[:, i].flatten(), 
                    multivariate_modeled[:, i].flatten())
        multivariate_r2.append(r2)
    
    return multivariate_r2

def metrics_transformer(truths, hats, phase, run):
    """
    Calculate the mean square error and mean absolute error
    and r2 for each variate of the model's predictions.
    ----------
    Arguments:
    truth (np.array): np.array, the observed values
    hat (np.array): the model's predictions
    phase (str): the phase of the data. Must be one of "train", "val", or "test"
    
    Returns:
    mse (list): root mean square error
    mae (list): mean absolute error
    r2 (list): r-squared value
    """

    # Define objects to store the results
    mse_final, mae_final, r2_final = [], [], []

    # Parse the truths and hats objects
    for tgt_y, y_hat in zip(truths, hats):

        mse = get_multivariate_mse(tgt_y, y_hat)
        mae = get_multivariate_mae(tgt_y, y_hat)
        r2 = get_multivariate_r2(tgt_y, y_hat)

        # Append results
        mse_final.append(mse)
        mae_final.append(mae)
        r2_final.append(r2)
    
    # Get the mean of each metric by variable
    mse_final, mae_final, r2_final = np.array(mse_final), np.array(mae_final), np.array(r2_final)
    mse_final = np.mean(mse_final, axis=0)
    mae_final = np.mean(mae_final, axis=0)
    r2_final = np.mean(r2_final, axis=0)

    # Create a dictionary with the metrics
    metrics = {
        'Mean squared error': mse_final,
        'Mean absolute error': mae_final,
        'R-squared': r2_final,
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(metrics, index=['am', 'co', 'do', 'ph', 'pr', 'tu', 'wt'])
    
    # Write the results to a text file
    results_path = f'results/run_t_{run}/results.txt'
    with open(results_path, 'a') as f:
        f.write(f'-- {phase} results\n')
        f.write(df.to_string())  # Convert DataFrame to a string and write it
        f.write('\n\n')

    # Close the file
    f.close()

    return mse_final, mae_final, r2_final

def metrics_unet(truth, hat, phase, run):
    """
    Calculate the root mean square error, and confusion matrix
    ----------
    Arguments:
    truth (np.array): np.array, the observed values
    hat (np.array): the model's predictions
    phase (str): the phase of the data. Must be one of "train", "val", or "test"
    
    Returns:
    rmse (float): root mean square error
    confusion_matrix (np.array): confusion matrix
    """
    
    rmse = np.sqrt(mean_squared_error(truth, hat))
    
    # Convert the predictions to binary values (0 or 1)
    truth = np.where(truth > 0.5, 1, 0)
    hat = np.where(hat > 0.5, 1, 0)
    
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(truth, hat)
    
    results_path = f'results/run_u_{run}/results.txt'
    with open(results_path, 'a') as f:
        f.write(f'-- {phase} results\n')
        f.write(f'Root mean square error: {rmse}\n')
        f.write(f'Confusion matrix:\n{confusion_matrix}\n')
        f.write('\n')

        # Close the file
        f.close()

    return rmse, confusion_matrix

def plots_transformer(date, src, truth, hat, weights, tgt_percentage, station, phase, instance):
    
    """
    Plot the observed and predicted values
    ----------
    Arguments:
    src (np.array): np.array, the src values before the prediction
    truth (np.array): np.array, the observed values during the prediction
    hat (np.array): the model's predictions
    weights (np.array): the weights of the model
    tgt_percentage (float): the percentage of the tgt values to plot
    multiple (int): the multiple of the tgt values to plot
    station (int): the station number
    phase (str): the phase of the data. Must be one of "train", "val", or "test"
    instance (int): the instance number

    Returns:
    None
    """

    # Subset the last row of the weights
    weights = weights[-1]
    
    # Concatenate the src with the tgt (truth) values 
    ground_truth = np.concatenate((src[-int(len(src)*tgt_percentage):], truth))
    
    # Plot the truth and the prediction
    colors_ground_truth = ['red', 'dodgerblue', 'mediumpurple', 'dimgrey', 'chocolate', 'goldenrod', 'green', 'black']
    colors_hat = ['salmon', 'lightskyblue', 'plum', 'darkgrey', 'orange', 'gold', 'yellowgreen', 'black']
    labels = ['am', 'co', 'do', 'ph', 'pr', 'tu', 'wt', 'lb']

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Plot bars on primary axes
    x_data = range(len(src))
    bars = ax1.bar(x_data, weights, color='darkseagreen', width=0.7, label='Weights')

    # Invert the y-axis for bars
    ax1.invert_yaxis()

    # Loop over each column of the truth
    for i in range(ground_truth.shape[1]):
        ax2.plot(ground_truth[:, i], label=f'{labels[i]}', color=colors_ground_truth[i], linewidth=0.5)

    # Loop over each column of the prediction and plot it starting when the concatenated part of tgt ends
    tgt_length = int(len(src)*tgt_percentage)
    for i in range(hat.shape[1]):
        ax2.plot(range(tgt_length, tgt_length + hat.shape[0]), hat[:, i], color=colors_hat[i], linewidth=0.5)

    plt.title(f'Results for instance {date.day}-{date.month}-{date.year}-{date.hour} of station {station}')
    plt.xlabel(r'time (15 min)')
    ax1.set_ylabel('Weights', color='dimgray')
    ax2.set_ylabel(r'Values', color='dimgray')
    
    bars, labels0 = ax1.get_legend_handles_labels()  # Get bars and labels for legend
    lines1, labels1 = ax2.get_legend_handles_labels()  # Get lines and labels for legend

    # Concatenate the bars and lines, and their respective labels
    handles = bars + lines1
    labels = labels0 + labels1

    # Add legend to primary axes
    plt.legend(handles, labels, loc='upper left')  # Add legend to primary axes
    
    # Show the plot
    plt.tight_layout()
    # plt.show()

    plt.savefig(f'plots/tplot_{date.day}_{date.month}_{date.year}_{date.hour}.png', dpi=300)

    # Close the plot
    plt.close(fig)

def plots_unet(truth, hat, station, phase, run):
    
    """
    Plot the observed and predicted values
    ----------
    Arguments:
    truth (np.array): np.array, the observed values
    hat (np.array): the model's predictions
    phase (str): the phase of the data. Must be one of "train", "val", or "test"

    Returns:
    None
    """

    # Plot predictions
    plt.figure();plt.clf()
    plt.plot(truth, label='observed', linewidth=0.5) # Transform into a confusion matrix and plot it as a heatmap
    plt.plot(hat, label='predicted', linewidth=0.5)
    plt.title(f'Station {station} {phase} results')
    plt.xlabel(r'time (15 min)')
    plt.ylabel(r'y')
    plt.legend()
    # plt.show()

    plt.savefig(f'results/run_y_{run}/u_{phase}.png', dpi=300)

def logger_transformer(run, batches, d_model, n_heads, encoder_layers, decoder_layers, dim_ll_encoder, dim_ll_decoder, lr, loss, epochs, seed, run_time):

    """Save the results of each run. The results
    are the hyperparameters of the model and the
    resulting plots.
    ----------
    Arguments:
    run (int): the run number defined by the user.
    
    Returns:
    None.
    """

    # Create a directory to save the results
    if not os.path.exists('results/run_t_{}'.format(run)):
        os.makedirs('results/run_t_{}'.format(run))
    
    # Save the hyperparameters in a text file
    with open('results/run_t_{}/results.txt'.format(run), 'w') as f:
        f.write('batches: ' + str(batches) + '\n')
        f.write('d_model: ' + str(d_model) + '\n')
        f.write('n_heads: ' + str(n_heads) + '\n')
        f.write('encoder_layers: ' + str(encoder_layers) + '\n')
        f.write('decoder_layers: ' + str(decoder_layers) + '\n')
        f.write('dim_ll_encoder: ' + str(dim_ll_encoder) + '\n')
        f.write('dim_ll_decoder: ' + str(dim_ll_decoder) + '\n')
        f.write('lr: ' + str(lr) + '\n')
        f.write('loss: ' + str(loss) + '\n')
        f.write('n_epochs: ' + str(epochs) + '\n')
        f.write('seed: ' + str(seed) + '\n')
        f.write('run_time: ' + str(run_time) + '\n')
        f.write('\n')

        # Close the file
        f.close()

def logger_unet(run, batches, input_channels, channels, d_fc, lr, loss, epochs, seed, run_time):

    """Save the results of each run. The results
    are the hyperparameters of the model and the
    resulting plots.
    ----------
    Arguments:

    Returns:
    """

        # Create a directory to save the results
    if not os.path.exists('results/run_u_{}'.format(run)):
        os.makedirs('results/run_u_{}'.format(run))
    
    # Save the hyperparameters in a text file
    with open('results/run_u_{}/results.txt'.format(run), 'w') as f:
        f.write('batches: ' + str(batches) + '\n')
        f.write('d_model: ' + str(input_channels) + '\n')
        f.write('input_channels: ' + str(input_channels) + '\n')
        f.write('channels: ' + str(channels) + '\n')
        f.write('d_fc: ' + str(d_fc) + '\n')
        f.write('lr: ' + str(lr) + '\n')
        f.write('loss: ' + str(loss) + '\n')
        f.write('n_epochs: ' + str(epochs) + '\n')
        f.write('seed: ' + str(seed) + '\n')
        f.write('run_time: ' + str(run_time) + '\n')
        f.write('\n')

        # Close the file
        f.close()