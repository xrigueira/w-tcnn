import pandas as pd

from tictoc import tictoc

# Smooths a column of data using a moving average with specified window size and stride
def smooth_column(column_data, window_size, stride):
    smoothed_values = []
    for i in range(0, len(column_data), stride):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(column_data), i + window_size // 2 + 1)
        window = column_data[window_start:window_end]
        smoothed_values.append(window.mean())
    return smoothed_values

@tictoc
def smoother(station):
    """This function normalizes and smoothes the data.
    ---------
    Arguments:
    station (str): the station name.

    Returns:
    smoothed_data (Pandas DataFrame): smoothed data."""

    # Read the data
    data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data.iloc[:, 1:-2] = scaler.fit_transform(data.iloc[:, 1:-2])

    # Define the variables needed for smoothing
    window_size = 4
    stride = 1
    smoothed_data = data.copy()

    # Smoothed the data using parallelization to speed it up
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        smoothed_columns = executor.map(smooth_column, 
                                        [data[col] for col in data.columns[1:-2]],
                                        [window_size] * (len(data.columns[1:-2])),
                                        [stride] * (len(data.columns[1:-2])))
        
        for col, smoothed_values in zip(data.columns[1:-2], smoothed_columns):
            smoothed_data[col] = smoothed_values

    smoothed_data.to_csv(f'data/labeled_{station}_smo.csv', encoding='utf-8', sep=',', index=False)
