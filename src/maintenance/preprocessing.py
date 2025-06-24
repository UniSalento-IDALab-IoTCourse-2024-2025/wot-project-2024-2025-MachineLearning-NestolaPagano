import pandas as pd
import numpy as np
import os
import random

SESSION_DURATION = 1200  # 20 minuti in secondi
AUGMENT_FACTOR   = 5

def augment_driving_data(df, augment_factor):
    """
    Prende un dataframe con colonne:
      ['Timestamp','AccX','AccY','AccZ','GyroX','GyroY','GyroZ','Class']
    e restituisce un df più grande
    """
    augmented_dfs = []
    sensor_cols = ['AccX','AccY','AccZ','GyroX','GyroY','GyroZ']

    for _ in range(augment_factor):
        augmented = df.copy().reset_index(drop=True)
        orig_len  = len(augmented)

        # rumore gaussiano
        noise_level = random.uniform(0.02, 0.08)
        noise = np.random.normal(
            loc=0,
            scale=noise_level,
            size=(orig_len, len(sensor_cols))
        )
        augmented[sensor_cols] = augmented[sensor_cols].values + noise

        # time warping (interpolazione su new_len)
        scale = random.uniform(0.85, 1.15)
        new_len = int(orig_len * scale)
        orig_x = np.arange(orig_len)
        new_x  = np.linspace(0, orig_len - 1, new_len)

        data_matrix = augmented[sensor_cols].values
        warped_matrix = np.zeros((new_len, 6), dtype=float)

        for j in range(6):
            warped_matrix[:, j] = np.interp(new_x, orig_x, data_matrix[:, j])

        if new_len >= orig_len:
            final_matrix = warped_matrix[:orig_len, :]
        else:
            pad_size  = orig_len - new_len
            last_row  = warped_matrix[-1, :]
            pad_block = np.tile(last_row[np.newaxis, :], (pad_size, 1))
            final_matrix = np.vstack([warped_matrix, pad_block])

        augmented[sensor_cols] = final_matrix

        # aggiorna uniformemente i timestamp
        t_min   = augmented['Timestamp'].min()
        t_max   = augmented['Timestamp'].max()
        t_range = t_max - t_min
        if orig_len > 1:
            new_timestamps = t_min + (np.arange(orig_len) / (orig_len - 1)) * t_range
            augmented['Timestamp'] = new_timestamps

        # flipping (inversione temporale 50%)
        if random.random() > 0.5:
            augmented = augmented.iloc[::-1].reset_index(drop=True)
            t_min   = augmented['Timestamp'].min()
            t_max   = augmented['Timestamp'].max()
            t_range = t_max - t_min
            if orig_len > 1:
                new_timestamps = t_min + (np.arange(orig_len) / (orig_len - 1)) * t_range
                augmented['Timestamp'] = new_timestamps

        # scaling di intensità
        scale_factor = random.uniform(0.8, 1.2)
        augmented[sensor_cols] = augmented[sensor_cols] * scale_factor

        augmented_dfs.append(augmented)

    return pd.concat(augmented_dfs, ignore_index=True)


def process_driving_data(path, path_2, session_duration, augment_factor):
    if not os.path.exists(path) or not os.path.exists(path_2):
        raise FileNotFoundError("Uno o entrambi i path di input non esistono")

    train_df = pd.read_csv(path)
    test_df  = pd.read_csv(path_2)
    full_df  = pd.concat([train_df, test_df], ignore_index=True)

    augmented_df = augment_driving_data(full_df, augment_factor)

    # Calcola la magnitudine di accelerometro e giroscopio
    augmented_df['accel_magnitude'] = np.sqrt(
        augmented_df['AccX']**2 +
        augmented_df['AccY']**2 +
        augmented_df['AccZ']**2
    )
    augmented_df['gyro_magnitude'] = np.sqrt(
        augmented_df['GyroX']**2 +
        augmented_df['GyroY']**2 +
        augmented_df['GyroZ']**2
    )

    # Calcola inizio/fine e numero totale di secondi
    min_timestamp  = augmented_df['Timestamp'].min()
    max_timestamp  = augmented_df['Timestamp'].max()
    total_duration = max_timestamp - min_timestamp

    # Determina quante sessioni da session_duration (o almeno 1)
    num_sessions = max(int(total_duration // session_duration), 1)

    session_rows = []
    for i in range(num_sessions):
        start_time = min_timestamp + i * session_duration
        end_time   = start_time + session_duration

        mask = (augmented_df['Timestamp'] >= start_time) & (augmented_df['Timestamp'] < end_time)
        session_df = augmented_df[mask]
        if session_df.empty:
            continue

        # Conteggi di ogni stile guida
        counts = session_df['Class'].value_counts()
        n_agg = int(counts.get('AGGRESSIVE', 0))
        n_nor = int(counts.get('NORMAL',    0))
        n_slo = int(counts.get('SLOW',      0))
        total_events = n_agg + n_nor + n_slo

        # Statistiche sensori
        accel_mag_mean = session_df['accel_magnitude'].mean()
        accel_mag_std  = session_df['accel_magnitude'].std()
        gyro_mag_mean  = session_df['gyro_magnitude'].mean()
        gyro_mag_std   = session_df['gyro_magnitude'].std()

        # Calcolo urgency
        if n_agg == 0:
            # Se non ci sono eventi aggressivi, mette urgenza zero
            urgency = 0.0
        else:
            # Proporzione di aggressività (da 0 a 1)
            aggressive_ratio = n_agg / total_events

            # Normalizza magnitudine accelerometro e giroscopio
            term_events = aggressive_ratio
            term_accel  = min(accel_mag_mean / 5.0, 1.0)
            term_gyro   = min(gyro_mag_mean  / 2.0, 1.0)

            # Formula pesata: 50% eventi + 30% accel + 20% gyro
            urgency = 0.5 * term_events + 0.3 * term_accel + 0.2 * term_gyro
            urgency = min(max(urgency, 0.0), 1.0)

        session_rows.append({
            'session_id':            f"session_{i}",
            'start_timestamp':       start_time,
            'end_timestamp':         end_time,
            'duration_minutes':      (end_time - start_time) / 60.0,
            'count_aggressive':      n_agg,
            'count_normal':          n_nor,
            'count_slow':            n_slo,
            'accel_mag_mean':        accel_mag_mean,
            'accel_mag_std':         accel_mag_std,
            'gyro_mag_mean':         gyro_mag_mean,
            'gyro_mag_std':          gyro_mag_std,
            'maintenance_urgency':   urgency
        })

    return pd.DataFrame(session_rows)


if __name__ == "__main__":
    path  = '../../data/motion_data.csv'
    path_2   = '../../data/motion_data_2.csv'
    output_path = '../../data/augmented_train_sessions.csv'

    try:
        sessions_df = process_driving_data(
            path, path_2, SESSION_DURATION, AUGMENT_FACTOR
        )
        sessions_df.to_csv(output_path, index=False)

        total_sessions = len(sessions_df)
        avg_aggressive = sessions_df['count_aggressive'].mean()
        avg_urgency    = sessions_df['maintenance_urgency'].mean()

        print("Elaborazione completata con successo!")
        print(f"Sessioni totali create         : {total_sessions}")
        print(f"Eventi aggressivi medi/sessione : {avg_aggressive:.2f}")
        print(f"Urgenza manutenzione media      : {avg_urgency:.2f}")
        print(f"Dataset salvato in              : {output_path}")

    except Exception as e:
        print("Errore durante l'elaborazione:", str(e))
