
import os
import pandas as pd

"""This function deletes those time spans across several variables 
with too many empty values, and iterates on the rest"""

def mfilterer(File, timeframe, timestep):

    fileName, fileExtension = os.path.splitext(File)
    df = pd.read_csv(f'data/{fileName}.csv', delimiter=',')

    years = list(dict.fromkeys(df['year'].tolist()))

    months = list(dict.fromkeys(df['month'].tolist()))
    months.sort()

    weeks = list(dict.fromkeys(df['week'].tolist()))
    weekOrder = list(dict.fromkeys(df['weekOrder'].tolist()))

    startDate = list(df['startDate'])
    endDate = list(df['endDate'])

    days = list(dict.fromkeys(df['day'].tolist()))

    cols = list(df.columns.values.tolist())[1:-11]

    if timestep == '15 min':
        limit_numNaN_a = 480
        limit_consecNaN_a = 192
        limit_numNaN_b = 192
        limit_consecNaN_b = 24
        limit_numNaN_c = 20
        limit_consecNaN_c = 12
        lenMonth = 2976
        lenWeek = 627

    elif timestep == '1 day':
        limit_numNaN_a = 7
        limit_consecNaN_a = 5
        limit_numNaN_b = 3
        limit_consecNaN_b = 2
        lenMonth = 31
        lenWeek = 7

    if timeframe == 'a':

        indexInit, indexEnd = [], []
        numNaN, consecNaN = [], []
        for i in years:

            df = df.loc[df['year'] == i]
            
            for j in months:
                
                df = df.loc[df['month'] == j]

                if df.empty == True:
                    pass
                elif df.empty == False:

                    # Get total number of NaN and the max consecutive NaNs
                    for k in cols:

                        numNaN.append(df[k].isnull().sum())
                        consecNaN.append(max(df[k].isnull().astype(int).groupby(df[k].notnull().astype(int).cumsum()).sum()))

                    # Count the number of NaN higher than 480 and the number of consecNaN higher than 192
                    count_numNaN = sum(map(lambda x: x >= limit_numNaN_a, numNaN))
                    count_consecNaN = sum(map(lambda x: x >= limit_consecNaN_a, consecNaN))
                    
                    # Get the first and last index of those months with too many empty (or consecutive) values 
                    # in several variables (NaN in this case)
                    if count_numNaN >= 3 or count_consecNaN >= 3:
                        indexInit.append(df.index[0])
                        indexEnd.append(df.index[-1])

                    # Clean numNaN and consecNaN
                    numNaN, consecNaN = [], []

                    df = pd.read_csv(f'data/{fileName}.csv', delimiter=',')
                    
                    if j == 12:
                        df = df.loc[df['year'] == (i+1)]
                    else:
                        df = df.loc[df['year'] == i]

        # Delete those parts of the data frame between the appended indices
        df = pd.read_csv(f'data/{fileName}.csv', delimiter=',')

        counter = 0
        # lenMonth = 2976
        for i,j in zip(indexInit, indexEnd):

            df = df.drop(df.index[int(i-counter*lenMonth):int(j-counter*lenMonth+1)], inplace=False)
            counter += 1
        
        # Interpolate the remaining empty values
        df = (df.interpolate(method='polynomial', order=1)).round(2)

        # Delete the columns needed for preprocessing
        df = df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second', 'startDate', 'endDate', 'weekOrder'])
        
        # Save the data frame
        cols = list(df.columns.values.tolist())
        df.to_csv(f'data/{fileName}_pro.csv', sep=',', encoding='utf-8', index=False, header=cols)

    elif timeframe == 'b':
        
        weeks = [i for i in weeks if i != 0] # Remove the 0s

        indexInit, indexEnd = [], []
        numNaN, consecNaN = [], []
        for i in weeks:
            
            df = df.loc[df['week'] == i]

            if df.empty == True:
                pass
            elif df.empty == False:
            
                # Get total number of NaN and the max consecutive NaNs
                for k in cols:
                    
                    numNaN.append(df[k].isnull().sum())
                    consecNaN.append(max(df[k].isnull().astype(int).groupby(df[k].notnull().astype(int).cumsum()).sum()))
                
                # Count the number of NaN higher than 192 and the number of consecNaN higher than 24
                count_numNaN = sum(map(lambda x: x >= limit_numNaN_b, numNaN))
                count_consecNaN = sum(map(lambda x: x >= limit_consecNaN_b, consecNaN))
                
                # Get the first and last index of those months with too many empty (or consecutive) values 
                # in several variables (NaN in this case)
                if count_numNaN >= 3 or count_consecNaN >= 3:
                    indexInit.append(df.index[0])
                    indexEnd.append(df.index[-1])
                
                # Clean numNaN and consecNaN
                numNaN, consecNaN = [], []
                
                df = pd.read_csv(f'data/{fileName}.csv', delimiter=',')
            
        # Delete those parts of the data frame between the appended indices
        df = pd.read_csv(f'data/{fileName}.csv', delimiter=',')
        
        counter = 0
        lenWeek = 672
        for i,j in zip(indexInit, indexEnd):
            
            df = df.drop(df.index[int(i-counter*lenWeek):int(j-counter*lenWeek+1)], inplace=False)
            counter += 1
        
        # Interpolate the remaining empty values
        df = (df.interpolate(method='polynomial', order=1)).round(2)
        
        # Delete the columns needed for preprocessing
        df = df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second', 'startDate', 'endDate', 'weekOrder'])

        # Save the data frame
        cols = list(df.columns.values.tolist())
        df.to_csv(f'data/{fileName}_pro.csv', sep=',', encoding='utf-8', index=False, header=cols)
        
    elif timeframe == 'c':

        indexInit, indexEnd = [], []
        numNaN, consecNaN = [], []
        
        for i in years:

            for j in months:

                for k in days:

                    filtered_df = df.loc[(df['year'] == i) & (df['month'] == j) & (df['day'] == k)]

                    if not filtered_df.empty:
                    
                        # Get total number of NaN and the max consecutive NaNs
                        for l in cols:

                            numNaN.append(filtered_df[l].isnull().sum())
                            consecNaN.append(max(filtered_df[l].isnull().astype(int).groupby(filtered_df[l].notnull().astype(int).cumsum()).sum()))

                        # Count the number of NaN higher than 24 and the number of consecNaN higher than 8
                        count_numNaN = sum(map(lambda x: x >= limit_numNaN_c, numNaN))
                        count_consecNaN = sum(map(lambda x: x >= limit_consecNaN_c, consecNaN))
                        
                        # Get the first and last index of those days with too many empty (or consecutive) values 
                        # in one variable (NaN in this case)
                        if count_numNaN >= 1 or count_consecNaN >= 1:
                            # print('here')
                            indexInit.append(filtered_df.index[0])
                            indexEnd.append(filtered_df.index[-1])
                        
                        # Clean numNaN and consecNaN
                        numNaN, consecNaN = [], []
        
        # Drop rows based on the filtered indices
        rows_to_drop = []
        for i, j in zip(indexInit, indexEnd):
            rows_to_drop.extend(range(i, j + 1))

        # Drop the rows using DataFrame.loc with inverse boolean indexing
        df = df.drop(df.index[rows_to_drop])

        # Interpolate the remaining empty values
        df = (df.interpolate(method='polynomial', order=2)).round(2)

        # Delete the columns needed for preprocessing
        df = df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second', 'startDate', 'endDate', 'weekOrder'])

        # Save the data frame
        cols = list(df.columns.values.tolist())
        df.to_csv(f'data/{fileName}_pro.csv', sep=',', encoding='utf-8', index=False, header=cols)
