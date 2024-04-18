import numpy as np


tgt_y_truth = np.load('tgt_y_truth.npy', allow_pickle=False, fix_imports=False)
tgt_y_hat = np.load('tgt_y_hat.npy', allow_pickle=False, fix_imports=False)


# TODO: Continue developing the plots here. There has to be one for each instance
# of the test dataset, and they have to include the last part (20%?) of tgt, and the full
# tgt_y_hat and tgt_y_truth. Therefore tgt_y_hat has to start where tgt ends, and there
# could tbe a vertical line at that point. Lasly, tgt_yhat could have the same colors
# as tgt_y_truth, but with a different style or lighter color.
