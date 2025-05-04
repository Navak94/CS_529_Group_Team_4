import sys
import os
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# === Validate Command Line Arguments ===
if len(sys.argv) != 3:
    print("Usage: python linear_regression_model.py <TICKER> <END_DATE>")
    sys.exit(1)

ticker = sys.argv[1].upper()
target_end_date = pd.to_datetime(sys.argv[2])  # converted to Timestamp

# === Load companies.json and find matching date range ===
with open("companies.json", "r") as file:
    config = json.load(file)

start_dates = config["start_date"]
end_dates = config["end_date"]

if sys.argv[2] not in end_dates:
    print(f"Error: End date {sys.argv[2]} not found in companies.json")
    sys.exit(1)

idx = end_dates.index(sys.argv[2])
start_date = start_dates[idx]
csv_filename = f"Example_output/{ticker}start_date_{start_date}_end_date{sys.argv[2]}.csv"

if not os.path.exists(csv_filename):
    print(f"Error: File not found for {ticker} on {sys.argv[2]}")
    sys.exit(1)

# === Load and Prepare Data ===
df = pd.read_csv(csv_filename, skiprows=1)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# === Train Model ===
X = df[['Close']]
y = df['Target']
model = LinearRegression()
model.fit(X, y)

# === Evaluate the Model ===
train_preds = model.predict(X)
mse = mean_squared_error(y, train_preds)
r2 = r2_score(y, train_preds)

# === Get 7 weekdays BEFORE end_date for prediction ===
available_dates = df.index[df.index < target_end_date]
recent_dates = available_dates[-7:]

comparison_rows = []
for date in recent_dates:
    actual = df.loc[date]['Close']
    predicted = model.predict(pd.DataFrame([[actual]], columns=["Close"]))[0]

    comparison_rows.append([date.date(), actual, predicted])

comparison_df = pd.DataFrame(comparison_rows, columns=["Date", "Actual", "Predicted"])
comparison_df.set_index("Date", inplace=True)

# === Save Output CSV ===
output_folder = "Prediction_output"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, f"{ticker}_predictions_{sys.argv[2]}.csv")
comparison_df.to_csv(output_path)

# === Print Result and Summary ===
print(comparison_df)
print(f"\nComparison saved to: {output_path}")

print("\nSummary:")
print(f"Company: {ticker}")
print(f"Data range used: {df.index.min().date()} to {df.index.max().date()}")
print(f"Training size: {len(df)}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
