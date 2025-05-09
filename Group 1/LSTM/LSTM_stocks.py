import tensorflow as tf
import json
import yfinance as yf
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Masking, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np


#hyperparameters from a lot of trial and error
HISTORICAL_YEARS = 10
MIN_REQUIRED_DAYS = 500
SEQUENCE_LENGTH = 5 #60
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.00001
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.1

#how far into the future we try to predict
PREDICTION_DAYS = 30

#########################################################
#           Reading from the Json file
#########################################################

# open our Json file in the group 1 folder
with open("Group 1/companies.json", "r") as file:
    data = json.load(file)

#make a list with all of the company symbols
companies = data["companies"]
start_date_json = data["start_date"]
end_date_json = data["end_date"]

#get the number of companies there are  in the Json file
num_of_companies = len(companies)

#getting the length of start dates.  i picked start date, since both start date and end date right now are the same length
date_count = len(start_date_json)

#########################################################
#               loop through each company
#########################################################
for i in range(date_count):
    for x in range(num_of_companies):

        #feed each company into the code we used earlier
        print(companies[x])

        # use the stock name we got from the Json file
        stock_symbol = companies[x]
        
        #pick a date to get data from
        start_date = start_date_json[i]
        end_date = end_date_json[i]

        # get the data from yfinance
        data = yf.download(stock_symbol, start=start_date, end=end_date)

        # im just putting all of these csv files in the examples folder 
        output_path = "Group 1/Example_output/" + companies[x] + "start_date_" +start_date_json[i] + "_" + "end_date" + end_date_json[i] + ".csv"

        data.to_csv(output_path)
        print(f"saved to " + companies[x] + "start_date_" +start_date_json[i] + "_" + "end_date" + end_date_json[i] + ".csv")

    #########################################################
    #           Fixing the formatting while we loop through
    #########################################################

        with open(output_path, "r") as file:
            lines = file.readlines()

        # fix the issue with putting "price" in the top left corner, put date there insead since its... you know..dates
        lines[0] = lines[0].replace("Price","Date") 

        # going to cut out the "ticker" line since we dont want that
        del lines[1]

        # have to do it again to remove the redundant empty "dates" row since we just deleted row 1
        # the other row we dont want is now the new row 1, so we delete row 1 again
        del lines[1]

        #let's put the stock name at the top of the file so it's easier for us to identify which stock we're looking at
        # this is OPTIONAL, but i do this to make things less confusing
        #lines.insert(0, "stock: " + stock_symbol + "\n")

        #re-write without the line we don't want and with that date fix
        with open(output_path, "w") as file:
            file.writelines(lines)

#######################################################################################
#                               Make the LSTM
#######################################################################################
# get the data form the close column from what we just made
        df = pd.read_csv(output_path)
        close_values = df["Close"].values.reshape(-1,1)
        scaler = MinMaxScaler()
        processed_data = scaler.fit_transform(close_values)


        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):

                X.append(data[i:i + seq_length])

                y.append(data[i + seq_length])

            return np.array(X), np.array(y)

        X, y = create_sequences(processed_data, SEQUENCE_LENGTH)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Define LSTM model
        def create_lstm_model():
            model = Sequential([
                Masking(mask_value=0.0, input_shape=(SEQUENCE_LENGTH, 1)),
                LSTM(LSTM_UNITS_1, return_sequences=True, kernel_regularizer=l2(0.001)),
                Dropout(DROPOUT_RATE),
                LSTM(LSTM_UNITS_2),
                Dropout(DROPOUT_RATE),
                Dense(1)
            ])
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'],run_eagerly=True)
            return model

        model = create_lstm_model()
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        
        # prediction making
        future_predictions = []
        input_seq = processed_data[-SEQUENCE_LENGTH:].tolist()

        # need to loop through our prediction days in this case, 30
        for _ in range(PREDICTION_DAYS):
            next_input = np.array(input_seq[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, 1)
            predicted = model.predict(next_input)
            input_seq.append(predicted[0])  # append prediction to sequence
            future_predictions.append(predicted[0])  # store for final output

        future_predictions = scaler.inverse_transform(future_predictions)
        print(future_predictions)


