import json
import yfinance as yf
import pandas as pd

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
        lines.insert(0, "stock: " + stock_symbol + "\n")

        #re-write without the line we don't want and with that date fix
        with open(output_path, "w") as file:
            file.writelines(lines)

