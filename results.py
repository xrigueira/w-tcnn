import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import utils

def plots_predictions(station: int = 901, run: int = 0, phase: str = 'test'):

    # Load transformer and UNet results object
    results_transformer = np.load(f'results/run_t_{run}/results_{phase}.npy', allow_pickle=True, fix_imports=True).item()  # Convert back to dict
    results_unet = np.load(f'results/run_u_{run}/results_{phase}.npy', allow_pickle=True, fix_imports=True).item()  # Convert back to dict

    # Plot results some dates
    plot_dates = pd.date_range(start=pd.Timestamp(2020, 11, 21, 0, 0, 0), 
                            end=pd.Timestamp(2020, 11, 27, 23, 0, 0), 
                            # end=pd.Timestamp(2020, 11, 21, 1, 0, 0), 
                            freq='h').to_list()

    for i, (date, data) in enumerate(results_transformer.items()):
        if date in plot_dates:
            utils.plots_transformer(date=date, src=data['src'], truth=data['tgt_y'], hat=data['y_hat'], weights=data['weights'], 
                                    tgt_percentage=1, output_sequence_len=4, station=station, phase=phase, 
                                    instance=date)

    scores = {}
    for i, (date, data) in enumerate(results_unet.items()):
        if date in plot_dates:
            scores[date] = {}
            scores[date]['tgt'] = data['tgt']
            scores[date]['y_hat'] = data['y_hat']

    # Save scores to a CSV file
    df = pd.DataFrame.from_dict(scores, orient='index')
    df.index.name = 'date'
    df.reset_index(inplace=True)
    df.to_csv(f'results/run_u_{run}/scores.csv', index=False)

def seasonal_weights_average(dates, results_transformer, phase: str, season: str):

    """Calculates the average weights for each season (winter, spring, summer, fall) of each year and smoothen the data.
    Args:
        dates: The dates range of the data.
        results_transformer: The dictionary containing the transformer results.
        season: The season to calculate the average weights for (winter, spring, summer, fall).
        anomalies: If True, calculate the average weights for the anomalies.
    Returns:
        weights_smooth (dict): The smoothened weights for each year."""
    
    # Define the seasons
    seasons = {
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'fall': [9, 10, 11]
    }

    # Get the months for the given season
    months = seasons[season]

    # Get the anomaly scores for the given season
    anomaly_dates = pd.date_range(start=pd.Timestamp(2020, 11, 26, 11, 0, 0), end=pd.Timestamp(2020, 11, 27, 11, 0, 0), freq='h').to_list() + pd.date_range(start=pd.Timestamp(2022, 8, 23, 18, 30, 0), end=pd.Timestamp(2022, 8, 23, 20, 30, 0), freq='h').to_list()

    # Define the arrays to store the weights for each season
    weights_average = [[] for _ in range(len(months))]
    
    # Extract weights for each month of the season
    for date in dates:
        if date.month in months: 
            month_index = months.index(date.month)
            weights_average[month_index].append(results_transformer[date]['weights'][-1])
    
    # Concatenate the weights for each month
    weights_average = [item for sublist in weights_average for item in sublist]
    
    # Average the weights for the season
    weights_average = np.mean(np.array(weights_average), axis=0)
    
    return np.array(weights_average)

def weights_analysis(station: int = 901, run: int = 0, phase: str = 'test'):
    
    def hex_to_rgb(hex_color):
        """Convert HEX to RGB."""
        return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

    def rgb_to_hex(rgb):
        """Convert RGB to HEX."""
        return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"

    def generate_gradient(start_hex, end_hex, length):
        """Generate a gradient of colors between start_hex and end_hex with the given length."""
        start_rgb = np.array(hex_to_rgb(start_hex))
        end_rgb = np.array(hex_to_rgb(end_hex))
        
        # Generate interpolated colors
        gradient = [
            rgb_to_hex(tuple(np.round(start_rgb + (end_rgb - start_rgb) * i / (length - 1)).astype(int)))
            for i in range(length)
        ]
        
        return gradient
    
    # Load transformer and UNet results object
    results_transformer = np.load(f'results/run_t_{run}/results_{phase}.npy', allow_pickle=True, fix_imports=True).item()  # Convert back to dict
    
    # Define the dates
    dates = list(results_transformer.keys())
    
    # Get the weights of each season
    winter_data = seasonal_weights_average(dates, results_transformer, phase=phase, season='winter')
    spring_date = seasonal_weights_average(dates, results_transformer, phase=phase, season='spring')
    summer_data = seasonal_weights_average(dates, results_transformer, phase=phase, season='summer')
    fall_data = seasonal_weights_average(dates, results_transformer, phase=phase, season='fall')
    
    # Create the plot
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(12, 6))

    # Plot the weights for each season# Plot the weights for each season with filled areas
    x_range = np.arange(-96, 0)

    # Winter plot
    axes[0, 0].fill_between(x_range, winter_data, alpha=0.3, color='#3a38a6')
    sns.lineplot(x=x_range, y=winter_data, color='#3a38a6', ax=axes[0, 0])

    # Spring plot
    axes[0, 1].fill_between(x_range, spring_date, alpha=0.3, color='#157d28')
    sns.lineplot(x=x_range, y=spring_date, color='#157d28', ax=axes[0, 1])

    # Summer plot
    axes[1, 0].fill_between(x_range, summer_data, alpha=0.3, color='#d99821')
    sns.lineplot(x=x_range, y=summer_data, color='#d99821', ax=axes[1, 0])

    # Fall plot
    axes[1, 1].fill_between(x_range, fall_data, alpha=0.3, color='#b53128')
    sns.lineplot(x=x_range, y=fall_data, color='#b53128', ax=axes[1, 1])

    # Set title for each subplot
    axes[0, 0].set_title('Winter', fontsize=16, fontname='Arial')
    axes[0, 1].set_title('Spring', fontsize=16, fontname='Arial')
    axes[1, 0].set_title('Summer', fontsize=16, fontname='Arial')
    axes[1, 1].set_title('Fall', fontsize=16, fontname='Arial')
    
    # Clean default y label and reduce font size for all axes
    for ax in axes.flat:
        ax.set_ylabel('Weights', fontsize=13, fontname='Arial')
        ax.set_xlabel('Time steps (15 min)', fontsize=13, fontname='Arial')
        ax.tick_params(axis='both', which='major', labelsize=10)

    # plt.show()

    # Save the plot
    plt.savefig(f'plots/attention_weights_{phase}.pdf', format="pdf", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    
    # Call the function to plot predictions
    # plots_predictions(station=901, run=0, phase='test')
    
    # Call the function to analyze weights
    weights_analysis(station=901, run=0, phase='test')
    