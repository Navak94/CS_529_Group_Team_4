import pandas as pd
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#########################################################
#           Getting the data from yfinance
#########################################################

# Define the stock symbol
stock_symbol = 'AAPL'

#get data between these dates (picking january 2023 as an example)
start_date = '2023-01-01'
end_date = '2023-01-31'

# download the data from yfinance
data = yf.download(stock_symbol, start=start_date, end=end_date)

# for some odd reason it wants to save to the folder above Group 1, just doing this to specify it should stay in our folder
output_path = "Group 1/Example_output/january_2023.csv"

# save the data to the csv file 
data.to_csv(output_path)

print(f"saved to january_2023.csv")


#########################################################
#           Fixing the formatting
#########################################################

# unfortunately yfinance kind of does a couple of things we dont like
# to begin with it saves the second row weird, for example we get a row that's 
# ticker, AAPL , AAPL , AAPL, AAPL
# personally i just find that annoying
# so to clean that up i usually just delete that row
# it also puts "price" in the spot where date should be, so usually i just do "replace"
# in addition to this it saves "date" underneath price, and it's kind of just alone in that row
# so this is all formatting stuff to fix that nonsense

# go read it all into lines
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

# cool!  now it's formatted correctly!
# you may wonder why January 1st and 2nd arent in the data, the reason why is because people weren't trading on those dates due to the holiday
# usually i target the "Close" column for stocks stuff

#########################################################################################################################################
#          Linear regression starts here
#########################################################################################################################################

#so we made a csv file with the data in it right, like the value of the stock and the date
# so we're going to read the csv (excel file) file we just made
df = pd.read_csv("Group 1/Example_output/january_2023.csv", parse_dates=["Date"])

# Turn dates into numbers (ordinal format)
df["DateOrdinal"] = df["Date"].map(pd.Timestamp.toordinal)

# we need to make x axis the date since it's time series data, and the x axis is usually time
dates_used = df["DateOrdinal"].values.reshape(-1, 1)

#specifically we want Y to be the close value for that day, so we're getting the close column in the spreadsheet.
close_value = df["Close"].values

#use the linear regression tool from the imports
model = LinearRegression()
model.fit(dates_used, close_value)
predictions = model.predict(dates_used)

#get the predictions
print ("predicted values")
print (predictions)

#find the most recent date in the csv file
data_end_date = df["Date"].max()

# i think we can make this anything just about but let's make it predict the trend for 1 week into the future
for i in range(1,8):# loop 7 times but start from tomorrow
    next_day_trend = data_end_date + pd.Timedelta(days=i)
    next_day_trend_ordinal = np.array([[next_day_trend.toordinal()]])
    next_prediction = model.predict(next_day_trend_ordinal)
    print(f"Predicted closing price for {next_day_trend.date()}: {next_prediction[0]:.2f}")
