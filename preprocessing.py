import os

from preprocessors.checkGaps import checkGaps
from preprocessors.normalizer import normalizer
from preprocessors.joiner import joiner
from preprocessors.filterer import mfilterer

"""Preprocessing starts from the original univariate txt files."""

# Define the data we want to study
files = [f for f in os.listdir("raw_data") if os.path.isfile(os.path.join("raw_data", f))]

# Extract the names of the variables
varNames = [i[0:-4] for i in files]

# Define the time dram we want to use (a: months (not recommended), b: weeks, c: days). 
timeFrame = 'b'
timeStep = '1 day'

if __name__ == '__main__':

    for varName in varNames:
    
        # Fill in the gaps in the time series
        checkGaps(File=f'{varName}.csv', timestep=timeStep, varname=varName)
        print('[INFO] checkGaps() DONE')

        # Normalize the data. See normalizer.py for details
        normalizer(File=f'{varName}_full.csv', timeframe=timeFrame, timestep=timeStep, varname=varName)
        print('[INFO] normalizer() DONE')
    
    # Join the normalized databases
    joiner(varNames)
    print('[INFO] joiner() DONE')

    # Filter out those months or weeks or days (depending on the desired
    # time unit) with too many NaN in several variables and iterate on the rest
    mfilterer(File=f'merged.csv', timeframe=timeFrame, timestep=timeStep)
    print('[INFO] filterer() DONE')