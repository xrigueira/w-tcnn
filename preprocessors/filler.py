
import os
import numpy as np
import pandas as pd
from datetime import datetime

"""There are two functions in this file. findMonday() returns the index of the first Monday in the data,
and filler() takes the filled data and adds the remaing days to set every month to 31 days.
This is done due to requierments of the functional data. In order to covert discrete data into functional
all time frames must have the same number of discrete values."""

# Function to get the index of the firts Monday
def findMonday(Dataframe):
    
    for i in range(len(Dataframe)):

        d = datetime(Dataframe['year'][i], Dataframe['month'][i], Dataframe['day'][i])
        
        if d.weekday() == 0:
            print('First Monday index: {} | {}'.format(i, d))
            break
    
    return i

def filler(File, timeframe, timestep, varname):

    if timestep == '15 min':
    
        fileName, fileExtension = os.path.splitext(File)
        df = pd.read_csv(f'data/{fileName}.csv', delimiter=',', parse_dates=['date'], index_col=['date'])

        # Add the needed columns (year, month, day, hour, min, sec)
        year = [i for i in df.index.year]
        month = [i for i in df.index.month]
        day = [i for i in df.index.day]
        hour = [i for i in df.index.hour]
        minute = [i for i in df.index.minute]
        second = [i for i in df.index.second]

        df['year'] = year
        df['month'] = month
        df['day'] = day
        df['hour'] = hour
        df['minute'] = minute
        df['second'] = second

        # Save temp file
        df.index.name = 'date'
        df.to_csv(f'data/{fileName}_temp.csv', sep=',', encoding='utf-8', index=True, header=[f'{varname}', 'year', 'month', 'day', 'hour', 'minute', 'second'])

        # Add the 31st day to those months with 30
        df = pd.read_csv(f'data/{fileName}_temp.csv', delimiter=',', parse_dates=['date'])

        if timeframe == 'a':

            monthShort = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if (x - y) == 29]
            print('These are the last indexes of the months with 30 days: ', monthShort)

            while monthShort:

                for i in monthShort:

                    if df['day'][i-1] == 30:
                        
                        startDate =  df.iloc[monthShort[0]-1, 0] 
                        yearInit, monthInit, dayInit, hourInit, minInit, secInit = startDate.year, startDate.month, startDate.day + 1, 0, 0, 0
                        rows = [[f'{yearInit}-{monthInit}-{dayInit} 00:00:00', np.nan, yearInit, monthInit, dayInit, hourInit, minInit, secInit]]

                        for j in range(95):

                            minInit += 15

                            if minInit > 45:

                                minInit = 0
                                hourInit += 1
                            
                            rows.append([f'{yearInit}-{monthInit}-{dayInit} {hourInit}:{minInit}:00', np.nan, yearInit, monthInit, dayInit, hourInit, minInit, secInit])

                        df2concat = pd.DataFrame(rows, columns=['date', f'{varname}', 'year', 'month', 'day', 'hour', 'minute', 'second'])

                        df = pd.concat([df.iloc[:i], df2concat, df.iloc[i:]], ignore_index=True)

                        day = df['day']
                        
                        monthShort = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if (x - y) == 29]

                        print('UPDATED indexes of months with 30 days: ', monthShort)

            # Add the 30th and 31st days to those February(s) on leap years:
            day = df['day']
            leaps = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if ((x - y) == 28 and df['month'][i-1] == 2)]
            print('These are the last indexes of the months with 29 days: ', leaps)

            if leaps: # Check if there are any leap years

                startDate =  df.iloc[leaps[0]-1, 0] 
                yearInit, monthInit, dayInit, hourInit, minInit, secInit = startDate.year, startDate.month, startDate.day + 1, 0, 0, 0
                rows = [[f'{yearInit}-{monthInit}-{dayInit} 00:00:00', np.nan, yearInit, monthInit, dayInit, hourInit, minInit, secInit]]

                while leaps:
                    
                    for i in leaps:
                        
                        if df['day'][i-1] == 29:
                            
                            startDate =  df.iloc[leaps[0]-1, 0] 
                            yearInit, monthInit, dayInit, hourInit, minInit, secInit = startDate.year, startDate.month, startDate.day + 1, 0, 0, 0
                            rows = [[f'{yearInit}-{monthInit}-{dayInit} 00:00:00', np.nan, yearInit, monthInit, dayInit, hourInit, minInit, secInit]]
                            
                            for j in range(191):
                                
                                minInit += 15

                                if hourInit == 23 and minInit > 45:
                                    
                                    minInit = 0
                                    hourInit = 0
                                    dayInit += 1

                                if minInit > 45:

                                    minInit = 0
                                    hourInit += 1
                                
                                rows.append([f'{yearInit}-{monthInit}-{dayInit} {hourInit}:{minInit}:00', np.nan, yearInit, monthInit, dayInit, hourInit, minInit, secInit])
                            
                            df2concat = pd.DataFrame(rows, columns=['date', f'{varname}', 'year', 'month', 'day', 'hour', 'minute', 'second'])

                            df = pd.concat([df.iloc[:i], df2concat, df.iloc[i:]], ignore_index=True)

                            day = df['day']
                            
                            leaps = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if ((x - y) == 28 and df['month'][i-1] == 2)]
                            
                            print('UPDATED indexes of months with 29 days: ', leaps)

            # Add the 29th, 30th, and 31st days to the February(s):
            day = df['day']
            febs = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if (x - y) == 27]
            print('These are the last indexes of the months with 28 days: ', febs)

            while febs:
                
                for i in febs:
                    
                    if df['day'][i-1] == 28:
                        
                        startDate =  df.iloc[febs[0]-1, 0] 
                        yearInit, monthInit, dayInit, hourInit, minInit, secInit = startDate.year, startDate.month, startDate.day + 1, 0, 0, 0
                        rows = [[f'{yearInit}-{monthInit}-{dayInit} 00:00:00', np.nan, yearInit, monthInit, dayInit, hourInit, minInit, secInit]]
                        
                        for j in range(287):
                
                            minInit += 15

                            if hourInit == 23 and minInit > 45:
                                
                                minInit = 0
                                hourInit = 0
                                dayInit += 1
                                
                            if minInit > 45:

                                minInit = 0
                                hourInit += 1
                            
                            rows.append([f'{yearInit}-{monthInit}-{dayInit} {hourInit}:{minInit}:00', np.nan, yearInit, monthInit, dayInit, hourInit, minInit, secInit])
                            
                        df2concat = pd.DataFrame(rows, columns=['date', f'{varname}', 'year', 'month', 'day', 'hour', 'minute', 'second'])

                        df = pd.concat([df.iloc[:i], df2concat, df.iloc[i:]], ignore_index=True)

                        day = df['day']
                        
                        febs = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if (x - y) == 27]
                        
                        print('UPDATED indexes of the months with 28 days: ', febs)

        # Store the index of the first Monday
        mondayIndex = findMonday(df)

        # Add three columns with the week number, start date, and end date, respectively
        weekIndex = []
        weekNumber = 0
        for i in range(mondayIndex):
            if i < mondayIndex:
                weekIndex.append(0)
                
        for i in range(len(df) - mondayIndex):
            
            if i % 672 == 0:
                weekNumber += 1
            weekIndex.append(weekNumber)

        df['week'] = weekIndex

        startDate = []
        for i in range(len(df['week'])):
            
            if i == 0:
                startDate.append('-')
            
            elif i > 0:
                
                if df['week'][i] != df['week'][i-1]:
                    
                    year, month, day = 'year', 'month', 'day'
                    dateS = f'{df[year][i]} {df[month][i]} {df[day][i]}'
                    startDate.append(dateS)
                else:
                    startDate.append('-')

        df['startDate'] = startDate

        endDate = []
        for i in range(len(df['week'])):
            
            if i < mondayIndex:
                endDate.append('-')
            
            elif i >= mondayIndex:
                
                if df['week'][i] != df['week'][i-1]:
                    if (i+672) < len(df['week']):
                        year, month, day = 'year', 'month', 'day' # Needed to avoid a syntax error
                        dateE = f'{df[year][i+671]} {df[month][i+671]} {df[day][i+671]}'
                        endDate.append(dateE)
                    else:
                        endDate.append('-')
                else:
                    endDate.append('-')

        df['endDate'] = endDate

        # Get the week order within every month
        weekOrder = []
        weekPosition = 0
        for i, e in enumerate(df['week']):
            
            if e == 0:
                weekOrder.append(0)
            
            else:
                if df['week'][i] == df['week'][i-1]:
                    weekOrder.append(weekPosition)
                elif df['week'][i] != df['week'][i-1]:
                    weekPosition += 1
                    if weekPosition == 5:
                        weekOrder.append(1)
                    else:
                        weekOrder.append(weekPosition)
                    if weekPosition == 5:
                        weekPosition = 1

        df['weekOrder'] = weekOrder

        # Save the new new file and remove the temp file
        cols = list(df.columns.values.tolist())
        df.to_csv(f'data/{fileName[0:-5]}_fil.csv', sep=',', encoding='utf-8', index=False, header=cols)
        os.remove(f'data/{fileName}_temp.csv')

    elif timestep == '1 day':
        
        fileName, fileExtension = os.path.splitext(File)
        df = pd.read_csv(f'data/{fileName}.csv', delimiter=',', parse_dates=['date'], index_col=['date'])
        
        # Add the needed columns (year, month, day)
        year = [i for i in df.index.year]
        month = [i for i in df.index.month]
        day = [i for i in df.index.day]

        df['year'] = year
        df['month'] = month
        df['day'] = day
        
        # Save temp file
        df.index.name = 'date'
        df.to_csv(f'data/{fileName}_temp.csv', sep=',', encoding='utf-8', index=True, header=[f'{varname}', 'year', 'month', 'day'])
        
        # Add the 31st day to those months with 30
        df = pd.read_csv(f'data/{fileName}_temp.csv', delimiter=',', parse_dates=['date'])
        
        if timeframe == 'a':

            monthShort = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if ((x - y) == 29)]
            # monthShort = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if ((x - y) == 29 or (i == (len(day) - 2) and y == 30))]
            print('These are the last indexes of the months with 30 days: ', monthShort)

            while monthShort:
                
                for i in monthShort:
                    
                    if df['day'][i-1] == 30:

                        startDate =  df.iloc[monthShort[0]-1, 0]
                        yearInit, monthInit, dayInit = startDate.year, startDate.month, startDate.day + 1
                        rows = [[f'{yearInit}-{monthInit}-{dayInit} 00:00:00', np.nan, yearInit, monthInit, dayInit]]
                        
                        df2concat = pd.DataFrame(rows, columns=['date', f'{varname}', 'year', 'month', 'day'])
                        
                        df = pd.concat([df.iloc[:i], df2concat, df.iloc[i:]], ignore_index=True)
                        
                        day = df['day']
                        
                        monthShort = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if ((x - y) == 29)]
                        
                        print('UPDATED indexes of months with 30 days: ', monthShort)
            
            # Add the 30th and 31st days to those February(s) on leap years:
            day = df['day']
            leaps = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if ((x - y) == 28 and df['month'][i-1] == 2)]
            print('These are the last indexes of the months with 29 days: ', leaps)
            
            if leaps: # Check if there are any leap years
                
                startDate =  df.iloc[leaps[0]-1, 0] 
                yearInit, monthInit, dayInit = startDate.year, startDate.month, startDate.day + 1
                rows = [[f'{yearInit}-{monthInit}-{dayInit} 00:00:00', np.nan, yearInit, monthInit, dayInit]]
                
                while leaps:
                    
                    for i in leaps:
                        
                        if df['day'][i-1] == 29:
                            
                            startDate =  df.iloc[leaps[0]-1, 0] 
                            yearInit, monthInit, dayInit = startDate.year, startDate.month, startDate.day + 1
                            rows = [[f'{yearInit}-{monthInit}-{dayInit} 00:00:00', np.nan, yearInit, monthInit, dayInit]]
                            
                            for j in range(1):
                                
                                rows.append([f'{yearInit}-{monthInit}-{(dayInit + (j + 1))} 00:00:00', np.nan, yearInit, monthInit, (dayInit + (j + 1))])

                            df2concat = pd.DataFrame(rows, columns=['date', f'{varname}', 'year', 'month', 'day'])
                            
                            df = pd.concat([df.iloc[:i], df2concat, df.iloc[i:]], ignore_index=True)

                            day = df['day']
                            
                            leaps = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if ((x - y) == 28 and df['month'][i-1] == 2)]
                            
                            print('UPDATED indexes of months with 29 days: ', leaps)
            
            # Add the 29th, 30th, and 31st days to the February(s):
            day = df['day']
            febs = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if (x - y) == 27]
            print('These are the last indexes of the months with 28 days: ', febs)
            
            while febs:
                
                for i in febs:
                    
                    if df['day'][i-1] == 28:
                        
                        startDate =  df.iloc[febs[0]-1, 0]
                        yearInit, monthInit, dayInit = startDate.year, startDate.month, startDate.day + 1
                        rows = [[f'{yearInit}-{monthInit}-{dayInit} 00:00:00', np.nan, yearInit, monthInit, dayInit]]
                        
                        for j in range(2):
                            
                            rows.append([f'{yearInit}-{monthInit}-{(dayInit + (j + 1))} 00:00:00', np.nan, yearInit, monthInit, (dayInit + (j + 1))])
                        
                        df2concat = pd.DataFrame(rows, columns=['date', f'{varname}', 'year', 'month', 'day'])

                        df = pd.concat([df.iloc[:i], df2concat, df.iloc[i:]], ignore_index=True)

                        day = df['day']
                        
                        febs = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if (x - y) == 27]
                        
                        print('UPDATED indexes of the months with 28 days: ', febs)

        # Store the index of the first Monday
        mondayIndex = findMonday(df)
        
        # Add three columns with the week number, start date, and end date, respectively
        weekIndex = []
        weekNumber = 0
        for i in range(mondayIndex):
            if i < mondayIndex:
                weekIndex.append(0)
                
        for i in range(len(df) - mondayIndex):
            
            if i % 7 == 0:
                weekNumber += 1
            weekIndex.append(weekNumber)

        df['week'] = weekIndex
        
        startDate = []
        for i in range(len(df['week'])):
            
            if i == 0:
                startDate.append('-')
            
            elif i > 0:
                
                if df['week'][i] != df['week'][i-1]:
                    
                    year, month, day = 'year', 'month', 'day'
                    dateS = f'{df[year][i]} {df[month][i]} {df[day][i]}'
                    startDate.append(dateS)
                else:
                    startDate.append('-')

        df['startDate'] = startDate
        
        endDate = []
        for i in range(len(df['week'])):
            
            if i < mondayIndex:
                endDate.append('-')
            
            elif i >= mondayIndex:
                
                if df['week'][i] != df['week'][i-1]:
                    if (i+7) <= len(df['week']):
                        year, month, day = 'year', 'month', 'day' # Needed to avoid a syntax error
                        dateE = f'{df[year][i+6]} {df[month][i+6]} {df[day][i+6]}'
                        endDate.append(dateE)
                    else:
                        endDate.append('-')
                else:
                    endDate.append('-')

        df['endDate'] = endDate

        # Get the week order within every month
        weekOrder = []
        weekPosition = 0
        for i, e in enumerate(df['week']):
            
            if e == 0:
                weekOrder.append(0)
            
            else:
                if df['week'][i] == df['week'][i-1]:
                    weekOrder.append(weekPosition)
                elif df['week'][i] != df['week'][i-1]:
                    weekPosition += 1
                    if weekPosition == 5:
                        weekOrder.append(1)
                    else:
                        weekOrder.append(weekPosition)
                    if weekPosition == 5:
                        weekPosition = 1

        df['weekOrder'] = weekOrder

        # Save the new new file and remove the temp file
        cols = list(df.columns.values.tolist())
        df.to_csv(f'data/{fileName[0:-5]}_fil.csv', sep=',', encoding='utf-8', index=False, header=cols)
        os.remove(f'data/{fileName}_temp.csv')
