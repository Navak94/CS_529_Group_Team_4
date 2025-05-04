# train_all_models.py

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# === Setup Paths ===
input_folder = "Example_output/"
model_output_folder = "Liner Regression/Trained_Models/"
summary_csv_path = "Liner Regression/results_summary.csv"
os.makedirs(model_output_folder, exist_ok=True)

# === To Store All Results ===
results = []

# === Loop through each CSV file ===
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        try:
            print(f"\n--- Processing: {filename} ---")

            # Load data
            df = pd.read_csv(file_path, skiprows=1)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # Feature Engineering
            df['SMA_3'] = df['Close'].rolling(window=3).mean()
            df['Volatility_3'] = df['Close'].rolling(window=3).std()

            df['Target'] = df['Close'].shift(-1)
            df.dropna(inplace=True)

            # Features: use Close, Volume, SMA, Volatility
            X = df[['Close', 'Volume', 'SMA_3', 'Volatility_3']]
            y = df['Target']

            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict and evaluate
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            last_row_scaled = scaler.transform([X.iloc[-1].values])
            next_day_prediction = model.predict(last_row_scaled)[0]

            # Save model
            model_name = filename.replace(".csv", "_model.joblib")
            joblib.dump(model, os.path.join(model_output_folder, model_name))

            # Store results
            results.append({
                "Company": filename.split("start_date")[0],
                "Start Date": filename.split("start_date_")[1].split("_end_date")[0],
                "End Date": filename.split("end_date")[1].replace(".csv", ""),
                "MSE": round(mse, 4),
                "R2": round(r2, 4),
                "Last Close": round(df.iloc[-1]['Close'], 2),
                "Next Day Prediction": round(next_day_prediction, 2)
            })

            print(f"  MSE: {mse:.4f} | R²: {r2:.4f}")
            print(f"  Last Close: ${df.iloc[-1]['Close']:.2f} | Prediction: ${next_day_prediction:.2f}")
            print(f"  Model saved as: {model_name}")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

# === Save Summary to CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv(summary_csv_path, index=False)
print(f"\n Results saved to {summary_csv_path}")

# === Plot R² and MSE per Company ===
plt.figure(figsize=(12, 5))
sns.barplot(data=results_df, x="Company", y="R2")
plt.title("R² Score per Company (with SMA + Volume)")
plt.ylabel("R² Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
sns.barplot(data=results_df, x="Company", y="MSE")
plt.title("MSE per Company (with SMA + Volume)")
plt.ylabel("Mean Squared Error")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
