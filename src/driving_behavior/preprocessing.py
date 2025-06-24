import numpy as np
import pandas as pd

def extract_window_features(df: pd.DataFrame, sampling_rate: int, window_duration_sec: int, step_duration_sec: int, label_threshold: float):

    df = df.copy()

    # Calcolo delle magnitudini
    df['Acc_Mag'] = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)
    df['Gyro_Mag'] = np.sqrt(df['GyroX']**2 + df['GyroY']**2 + df['GyroZ']**2)

    # Conversione dei secondi in numero di righe
    window_size = window_duration_sec * sampling_rate
    step = step_duration_sec * sampling_rate

    # Elenco delle colonne da cui estrarre le feature
    sensor_cols = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Acc_Mag', 'Gyro_Mag']
    features_list = []
    labels_list = []

    n_rows = len(df)
    # Scorrimento del dataframe
    for start in range(0, n_rows - window_size + 1, step):
        window = df.iloc[start : start + window_size]

        # Calcolo di media, deviazione, minimo, massimo per ogni colonna
        window_feats = []
        for col in sensor_cols:
            data_arr = window[col].values
            window_feats.extend([
                data_arr.mean(),
                data_arr.std(),
                data_arr.min(),
                data_arr.max()
            ])

        # Etichetta la finestra se una classe supera la soglia
        label_counts = window['Class'].value_counts(normalize=True)
        top_label = label_counts.idxmax()
        if label_counts.max() >= label_threshold:
            features_list.append(window_feats)
            labels_list.append(top_label)
        # se la percentuale della classe dominante è sotto threshold, si scarta

    X_windows = np.array(features_list)
    y_windows = np.array(labels_list)
    return X_windows, y_windows