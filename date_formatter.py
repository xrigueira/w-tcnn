"""This proram reads the raw data and puts the date column in the desired format.
I had to use this program because sometimes Excel did not read the date column correctly."""

import pandas as pd

# Read the data
data = pd.read_csv('raw_data/precipitation_A089_15min.txt', sep=';')
print(data.head())

# Convert the 'date' column to the desired format

data['Fecha'] = pd.to_datetime(data['Fecha'], format='%d/%m/%Y %H:%M').dt.strftime('%d-%m-%Y %H:%M:%S')

# Remove the 'Tag' column
data = data.drop('Tag', axis=1)

# Rename 'Fecha' to 'date'
data = data.rename(columns={'Fecha': 'Date'})
data = data.rename(columns={'valor': 'Value'})

print(data.head())

# Save the data as CSV wihout the index
data.to_csv('raw_data/precipitation_A089.txt', index=False)
