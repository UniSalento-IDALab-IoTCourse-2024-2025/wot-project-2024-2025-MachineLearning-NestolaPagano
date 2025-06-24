import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

from preprocessing import extract_window_features

def main():
    data = pd.read_csv('../../data/motion_data.csv')

    print("Encoding delle classi di guida in numeri interi")
    print("   - Classi originali:", data['Class'].unique())
    mapping = {'AGGRESSIVE': 1, 'NORMAL': 2, 'SLOW': 3}
    data['Class'] = data['Class'].map(mapping)
    print("   - Classi mappate:", data['Class'].unique(), "\n")

    # Estrazione feature a finestre mobili
    sampling_rate = 2             # 2 campioni al secondo
    window_duration_sec = 10      # 10 s → 20 righe
    step_duration_sec = 1         # 1 s → 2 righe
    label_threshold = 0.65        # 65%

    print("Estrazione delle feature a finestre mobili")
    X_win, y_win = extract_window_features(
        df=data,
        sampling_rate=sampling_rate,
        window_duration_sec=window_duration_sec,
        step_duration_sec=step_duration_sec,
        label_threshold=label_threshold
    )
    print(f"   - Numero di finestre estratte: {X_win.shape[0]}")
    print(f"   - Shape di X_win: {X_win.shape}, shape di y_win: {y_win.shape}")

    print("Train/Test split 80%/20% sulle finestre")
    X_train_win, X_test_win, y_train_win, y_test_win = train_test_split(
        X_win, y_win, test_size=0.2, random_state=42, stratify=y_win
    )
    print(f"   - X_train_win: {X_train_win.shape}, y_train_win: {y_train_win.shape}")
    print(f"   - X_test_win : {X_test_win.shape}, y_test_win : {y_test_win.shape}\n")

    print("Bilanciamento classi su training set con SMOTE")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_win, y_train_win)
    unique_res, counts_res = np.unique(y_train_res, return_counts=True)
    print(f"   - Nuova distribuzione classi (train bilanciato): {dict(zip(unique_res, counts_res))}\n")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    rf.fit(X_train_res, y_train_res)

    print("Valutazione sul test set")
    y_pred_win = rf.predict(X_test_win)
    acc_win = accuracy_score(y_test_win, y_pred_win)
    print(f"   - Accuracy sul test set: {acc_win:.4f}")
    print("   - Classification Report:")
    print(classification_report(
        y_test_win,
        y_pred_win,
        target_names=['AGGRESSIVE','NORMAL','SLOW']
    ))
    print("   - Confusion Matrix:")
    cm = confusion_matrix(y_test_win, y_pred_win)
    print(cm, "\n")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['AGG','NORM','SLOW'],
                yticklabels=['AGG','NORM','SLOW'])
    plt.title('Confusion Matrix - Random Forest su finestre')
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.tight_layout()
    plt.show()

    feature_columns = []
    for base in ['AccX','AccY','AccZ','GyroX','GyroY','GyroZ','Acc_Mag','Gyro_Mag']:
        feature_columns += [
            f"{base}_mean", f"{base}_std", f"{base}_max", f"{base}_min", f"{base}_range",
            f"{base}_q1", f"{base}_q3", f"{base}_iqr", f"{base}_zero_cross",
            f"{base}_autocorr", f"{base}_energy", f"{base}_dominant_freq"
        ]
    feature_columns.append('AccMag_GyroMag_corr')

    joblib.dump({
        'model': rf,
        'sampling_rate': sampling_rate,
        'window_duration_sec': window_duration_sec,
        'step_duration_sec': step_duration_sec,
        'feature_columns': feature_columns,
        'class_mapping': {1: 'AGGRESSIVE', 2: 'NORMAL', 3: 'SLOW'}
    }, '../../models/rf_driving_behavior_windows.joblib')
    print("Modello salvato in '../../models/rf_driving_behavior_windows.joblib'\n")

if __name__ == "__main__":
    main()