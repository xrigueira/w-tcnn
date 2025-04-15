import numpy as np
import pandas as pd

import utils

# Define variables
station = 901
run = 0

# Load transformer and UNet results object
phase = 'test'
results_transformer = np.load(f'results/run_t_{run}/results_{phase}.npy', allow_pickle=True, fix_imports=True).item()  # Convert back to dict
results_unet = np.load(f'results/run_u_{run}/results_{phase}.npy', allow_pickle=True, fix_imports=True).item()  # Convert back to dict

# Plot results some dates
plot_dates = pd.date_range(start=pd.Timestamp(2020, 11, 21, 0, 0, 0), 
                        end=pd.Timestamp(2020, 11, 27, 23, 0, 0), 
                        freq='h').to_list()

# for i, (date, data) in enumerate(results_transformer.items()):
#     if date in plot_dates:
#         utils.plots_transformer(date=date, src=data['src'], truth=data['tgt_y'], hat=data['y_hat'], weights=data['weights'], 
#                                 tgt_percentage=1, station=station, phase=phase, 
#                                 instance=date)

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