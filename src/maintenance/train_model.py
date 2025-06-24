import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def main():
    df = pd.read_csv('../../data/augmented_train_sessions.csv')

    features = [
        'count_aggressive', 'count_normal', 'count_slow',
        'duration_minutes',
        'accel_mag_mean', 'accel_mag_std',
        'gyro_mag_mean',  'gyro_mag_std'
    ]
    X = df[features]
    y = df['maintenance_urgency']

    # Split 80%/20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Valutazione
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"Test MAE : {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²  : {r2:.4f}")

    joblib.dump(model, '../../models/rf_maintenance_regressor.joblib')
    print("Modello salvato in '../../models/rf_maintenance_regressor.joblib'")

if __name__ == "__main__":
    main()