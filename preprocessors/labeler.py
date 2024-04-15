import numpy as np
import pandas as pd
from datetime import datetime

"""This function is used to label the merged or filled files
of the selected stations"""

def labeler(stations):

    # Read the anomalies file
    df_anomalies = pd.read_csv(f'anomalies.csv', sep=';', encoding='utf-8')

    # Group rows by station number and apply a lambda function to create tuples of start and end dates
    grouped = df_anomalies.groupby('Station').apply(lambda x: [(s, e) for s, e in zip(x.Start_date, x.End_date)])

    # Convert the resulting Series to a dictionary
    anomalies = grouped.to_dict()

    for station in anomalies.keys():
        # Only process the station if it's in the list of stations to process
        if station in stations:
            # Read the database, which can be either the merged or normalized
            df = pd.read_csv(f'data/merged_{station}.csv', sep=',', encoding='utf-8', parse_dates=['date'])

            # Add the label column with all zeros for now
            df['label'] = [0] * len(df)
            
            for dates in (anomalies[station]):
                startDate = datetime.strptime(dates[0], '%d-%m-%Y %H:%M:%S')
                endDate = datetime.strptime(dates[1], '%d-%m-%Y %H:%M:%S')

                # Update the value of the label
                df.loc[(df.date >= startDate) & (df.date <= endDate), 'label'] = 1

            # Save the database
            df.to_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8', index=False)


