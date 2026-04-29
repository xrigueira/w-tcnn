import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import utils

def build_features_labels(results_dict, label_series, label_threshold):
    """
    Parameters
    ----------
    results_dict   : transformer results dict {timestamp: {tgt_y, y_hat, ...}}
    label_series   : pd.Series with DatetimeIndex, values are 0/1 labels
    label_threshold: fraction threshold to binarise the label

    Returns
    -------
    X          : np.ndarray, shape (N, 56)
    y          : np.ndarray, shape (N,) — int labels
    timestamps : list of N Timestamps
    """
    timestamps = list(results_dict.keys())
    n = len(timestamps)

    # Stack all tgt_y and y_hat at once (N, 4, 7)
    tgt_y_all = np.array([results_dict[ts]['tgt_y'] for ts in timestamps], dtype=np.float32).reshape(n, -1)   # (N, 28)
    y_hat_all = np.array([results_dict[ts]['y_hat'] for ts in timestamps], dtype=np.float32).reshape(n, -1)   # (N, 28)
    X = np.concatenate([tgt_y_all, y_hat_all], axis=1)      # (N, 56)

    # Build labels: for each window the 4 predicted steps are ts+0,+15,+30,+45 min
    labels = np.empty(n, dtype=np.float32)
    label_index_set = set(label_series.index)
    for i, ts in enumerate(timestamps):
        step_labels = []
        for k in range(4):
            t = ts + k * timedelta
            if t in label_index_set:
                step_labels.append(label_series[t])
        if step_labels:
            labels[i] = float(np.mean(step_labels) > label_threshold)
        else:
            labels[i] = np.nan

    # Drop windows with no matching labels
    valid = ~np.isnan(labels)
    return X[valid], labels[valid].astype(int), [timestamps[i] for i in range(n) if valid[i]]

run = 1
station = 901
threshold = 0.5
timedelta = pd.Timedelta(minutes=15)

# Load transformer results (all splits)
transformer_train = np.load(f'results/run_t_{run}/results_train.npy', allow_pickle=True, fix_imports=True).item()
transformer_validation = np.load(f'results/run_t_{run}/results_validation.npy', allow_pickle=True, fix_imports=True).item()
transformer_test = np.load(f'results/run_t_{run}/results_test.npy', allow_pickle=True, fix_imports=True).item()

# Combine ALL transformer results into one pool
transformer_all = {**transformer_train, **transformer_validation, **transformer_test}

# Load data
data = utils.read_data(station=station, timestamp_col_name='date')
label_series = data['label']

# Build features and labels from the full pool
X_all, y_all, ts_all = build_features_labels(transformer_all, label_series, threshold)
print(f"All samples (before rebalancing) : {len(X_all)}  (anomalies: {int(y_all.sum())})")

# Rebalance: keep all anomalies + 12x background samples
rng = np.random.default_rng(seed=0)
anom_idx = np.where(y_all == 1)[0]
bg_idx   = np.where(y_all == 0)[0]
n_bg_keep = min(len(anom_idx) * 12, len(bg_idx))
bg_sampled = rng.choice(bg_idx, size=n_bg_keep, replace=False)
keep = np.sort(np.concatenate([anom_idx, bg_sampled]))
X_bal = X_all[keep]
y_bal = y_all[keep]
ts_bal = [ts_all[i] for i in keep]
print(f"All samples (after  rebalancing) : {len(X_bal)}  (anomalies: {int(y_bal.sum())})")

# Stratified 90/10 train/test split
X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(
    X_bal, y_bal, ts_bal,
    test_size=0.10, random_state=0, stratify=y_bal
)
print(f"RF train samples : {len(X_train)}  (anomalies: {int(y_train.sum())})")
print(f"RF test  samples : {len(X_test)}   (anomalies: {int(y_test.sum())})")

# Train Random Forest
rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=0, n_jobs=-1)
rf.fit(X_train, y_train)

# Print confusion matrix on train set
y_train_hat = rf.predict(X_train)
print("\n--- Train results ---")
print(f"Accuracy : {accuracy_score(y_train, y_train_hat):.4f}")
cm_train = confusion_matrix(y_train, y_train_hat)
print("Confusion matrix (rows=true, cols=pred):")
print(f"  TN={cm_train[0,0]}  FP={cm_train[0,1]}")
print(f"  FN={cm_train[1,0]}  TP={cm_train[1,1]}")

# Save model
pickle.dump(rf, open(f'results/run_t_{run}/rf.sav', 'wb'))
print(f"Model saved to results/run_t_{run}/rf.sav")

# Evaluate on test set
y_hat = rf.predict(X_test)

print("\n--- Test results ---")
print(f"Accuracy : {accuracy_score(y_test, y_hat):.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_hat, target_names=['Normal', 'Anomaly'], zero_division=0))

cm = confusion_matrix(y_test, y_hat)
print("Confusion matrix (rows=true, cols=pred):")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# Save predictions with timestamps
results_df = pd.DataFrame({
    'timestamp': ts_test,
    'y_true':    y_test.astype(int),
    'y_pred':    y_hat.astype(int)
})

results_df.to_csv(f'results/run_t_{run}/rf_predictions_test.csv', index=False)
print(f"\nPredictions saved to results/run_t_{run}/rf_predictions_test.csv")
