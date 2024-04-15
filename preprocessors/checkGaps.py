
import os
import numpy as np
import pandas as pd
from datetime import datetime

def checkGaps(File, timestep, varname):

    """The function checkGaps() look for gaps in the times series and fills them with the missing dates"""
    
    # Read the file
    fileName, fileExtension = os.path.splitext(File)

    if timestep == '15 min':
        df = pd.read_csv(f'raw_data/{fileName}.txt', delimiter=';', parse_dates=['Date'], date_format='%d-%m-%Y %H:%M:%S', index_col=['Date'])
        frequency = '15min'

    elif timestep == '1 day':
        df = pd.read_csv(f'raw_data/{fileName}.txt', delimiter=';', parse_dates=['Date'], date_format='%d-%m-%Y', index_col=['Date'])
        frequency = 'D'

    # Remove the index duplication
    df = df.loc[~df.index.duplicated(), :]

    # Check for missing dates
    df.index = pd.to_datetime(df.index)
    missingDates = pd.date_range(start=str(df.index[0]), end=str(df.index[-1]), freq=frequency).difference(df.index)

    # Get all the whole date range
    allDates = pd.date_range(start=str(df.index[0]), end=str(df.index[-1]), freq=frequency)

    #  Insert all dates in the db
    df = df.reindex(allDates, fill_value=np.nan)
    df.index.name = 'date'

    # Calculate the percentage of missing values in the 'precipitation_901' column
    missing_percentage = df['Value'].isnull().mean() * 100
    print(f'Percentage of missing values {varname}:', round(missing_percentage, ndigits=1), '%')
    
    # Save the db to csv
    df.to_csv(f'data/{fileName}_full.csv', sep=',', encoding='utf-8', index=True, header=[f'{varname}'])

