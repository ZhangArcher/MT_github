from yfinance import Ticker
import pandas as pd
import os

"""
  collect the stock price data using yfinance  
  
  Attributes
  ----------

  Methods:
  -------
  
"""


def download_data(stock_id_list:list, interval:str,start:str,end:str,
                     target_diretory:str ,auto_adjust:bool,back_adjust=bool):
    """
     download the stock price data using yfinance according stock id ,time interval ,time range
     and store in the target folder

      :arg
      -------
        stock_id_list: list
            the directory of the excel table
        interval:str
            data interval (intraday data cannot extend last 60 days)
            Valid intervals are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        start:str
            start date string
        end:str
            end date string
        target_diretory:str
            The target folder to store the result

      :return
      -------
        None
    """

    # download the stock price data for each stock id
    for stock_id in stock_id_list:
        #get stock prices
        history = Ticker(stock_id).history(interval=interval,
                                        start=start,end=end,
                                        auto_adjust=auto_adjust,back_adjust=back_adjust)

#       #set the file name
        file_name=stock_id+".csv"
        #save the data in the target folder
        history.to_csv(os.path.join(target_diretory,file_name))

def get_stock_id_from_excel(file_dir:str):
    """
      collect the stock id from excel and return a id list

      :arg
      -------
        file_dir: str
            the directory of the excel table


      :return
      -------
        stock_id_list: list[str]
            a list contains stock id
    """

    try:
        index_list = (pd.read_excel(file_dir))
        stock_id_list = index_list['Holding Ticker'].to_list()
        print("asdsd")
    except  Exception:
        print("can not find such table")

    return stock_id_list

if __name__ == '__main__':
    file_dir="stock_index_list.xls"
    stock_id_list=get_stock_id_from_excel(file_dir)
    #print(stock_id_list)
    download_data(stock_id_list,interval="1mo",start="1999-01-01",
                  end="2020-01-01",
                  target_diretory="long_1mo_with_back_ADJ",
                  back_adjust=True,auto_adjust=True)
