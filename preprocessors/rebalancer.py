import pandas as pd

def rebalancer(station):
    """This function creates a balanced dataset by selecting days with label == 1 and including 5 days before and after.
    ----------
    Arguments:
    station (str): the station name.

    Returns:
    balanced_data (Pandas DataFrame): balanced dataset.
    """

    # Read the data
    data = pd.read_csv(f'data/labeled_{station}_smo.csv', sep=',', parse_dates=['date'], encoding='utf-8')

    # Extract unique days
    data['day'] = data['date'].dt.date  # Extract only the date part

    # Identify days where at least one row has label == 1
    days_with_label_1 = set(data[data['label'] == 1]['day'])

    # Expand to include 5 days before and after each identified day
    selected_days = set()
    for day in days_with_label_1:
        for offset in range(-5, 6):  # -5 to +5 days
            selected_days.add(day + pd.Timedelta(days=offset))
    
    # Filter the dataset to keep only selected days
    filtered_data = data[data['day'].isin(selected_days)].drop(columns=['day'])

    # Save the balanced dataset
    filtered_data.to_csv(f'data/labeled_{station}.csv', encoding='utf-8', sep=',', index=False)
