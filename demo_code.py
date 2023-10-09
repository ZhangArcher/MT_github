
import data_set_price.data_collect as data_collect
import gp_wrapper_prices as gp_prices
import fusion_wrapper_price
import util

if __name__ == '__main__':
    #step 0 :to collect price data
    # collect the daily price data and save it in directory "demo_data/long" using yfiannce

    stock_id_list=["AAPL","NVDA","TSLA"]
    data_collect.download_data(stock_id_list,interval="1mo",start="1999-01-01",
                  end="2020-01-01",
                  target_diretory="demo_data/long",
                  back_adjust=True,auto_adjust=True)

  # collect the monthly price data and save it in directory "demo_data/short" using yfiannce
  #   data_collect.download_data(stock_id_list,interval="1d",start="1999-01-01",
  #                 end="2020-01-01",
  #                 target_diretory="demo_data/short",
  #                 back_adjust=True,auto_adjust=True)

  #  step 1: predict price using GP
    #  predict the monthly price of "AAPL" at 2015-02-01 using GP
    # GP is fitting by data set between [2010-01-01,2015-01-01]
    start_time = "2010-01-01 00:00:00"
    end_time = "2015-01-01 00:00:00"
    start = util.convert_time_into_datetime(time=start_time)
    end = util.convert_time_into_datetime(time=end_time)

    gp_price_AAPL=gp_prices.gp_wrapper_price(csv_file="demo_data/long/AAPL.csv")
    X_pred_times,Y_pred_mean,Y_pred_cov,Y_pred_actual,loss_score=gp_price_AAPL.predict(start_time=start_time,end_time=end_time,pred_length=1)
    #X_pred_times is timestamp [2015-02-01]
    #Y_pred_mean is the prediction mean
    #loss_score is the evaluation.  A perfect prediction means that loss_score is 0
    print("step1 :to predict the AAPL stock price on 01.02.2015 , while fiiting GP with the data set [01-01-2010,01-01-2015]")
    print("The actual price  on 01.02.2015 :",Y_pred_actual)
    print("The predicted price on 01.02.2015:",Y_pred_mean)
    print("gp_price_AAPL loss is :",loss_score)

    #  step 2: predict price using fusion GP
    #  predict the monthly price of "AAPL" at 2015-02-01 using GP
    # GP is fitting by data set between [2010-01-01,2015-01-01]


    fusion_price_AAPL = fusion_wrapper_price.fusion_wrapper_price(csv_path_long_term="demo_data/long/AAPL.csv",
                                                              csv_path_short_term="demo_data/short/AAPL.csv")

    X_time,Y_mean_fusion,Y_pred_mean_l,Y_pred_actual_l,error_fusion,error_l=fusion_price_AAPL.fusion_next_price(start=start,end=end,excess_time=15)

    # X_pred_times is timestamp [2015-02-01]
    # Y_pred_mean is the prediction mean
    # loss_score is the evaluation.  A perfect prediction means that loss_score is 0
    print("step2 :to predict the AAPL stock price on 01.02.2015 using fusion , while fiiting GP with the data set [01-01-2010,01-01-2015]")
    print("The actual price  on 01.02.2015 :", Y_pred_actual_l)
    print("The fused predicted price on 01.02.2015:", Y_mean_fusion)
    print("gp_price_AAPL loss is :", error_fusion)


  #  step 1: predict price using GP
    #  predict the monthly price of "AAPL" at 2015-02-01 using GP
    # GP is fitting by data set between [2010-01-01,2015-01-01]
    start_time = "2014-01-01 00:00:00"
    end_time = "2015-01-01 00:00:00"
    start = util.convert_time_into_datetime(time=start_time)
    end = util.convert_time_into_datetime(time=end_time)

    gp_price_AAPL=gp_prices.gp_wrapper_price(csv_file="demo_data/long/AAPL.csv")
    X_pred_times,Y_pred_mean,Y_pred_cov,Y_pred_actual,loss_score=gp_price_AAPL.predict(start_time=start_time,end_time=end_time,pred_length=1)
    #X_pred_times is timestamp [2015-02-01]
    #Y_pred_mean is the prediction mean
    #loss_score is the evaluation.  A perfect prediction means that loss_score is 0
    print("step1 :to predict the AAPL stock price on 01.02.2015 , while fiiting GP with the data set [01-01-2014,01-01-2015]")
    print("The actual price  on 01.02.2015 :",Y_pred_actual)
    print("The predicted price on 01.02.2015:",Y_pred_mean)
    print("gp_price_AAPL loss is :",loss_score)

    #  step 2: predict price using fusion GP
    #  predict the monthly price of "AAPL" at 2015-02-01 using GP
    # GP is fitting by data set between [2010-01-01,2015-01-01]


    fusion_price_AAPL = fusion_wrapper_price.fusion_wrapper_price(csv_path_long_term="demo_data/long/AAPL.csv",
                                                              csv_path_short_term="demo_data/short/AAPL.csv")

    X_time,Y_mean_fusion,Y_pred_mean_l,Y_pred_actual_l,error_fusion,error_l=fusion_price_AAPL.fusion_next_price(start=start,end=end,excess_time=15)

    # X_pred_times is timestamp [2015-02-01]
    # Y_pred_mean is the prediction mean
    # loss_score is the evaluation.  A perfect prediction means that loss_score is 0
    print("step2 :to predict the AAPL stock price on 01.02.2015 using fusion , while fiiting GP with the data set [01-01-2014,01-01-2015]")
    print("The actual price  on 01.02.2015 :", Y_pred_actual_l)
    print("The fused predicted price on 01.02.2015:", Y_mean_fusion)
    print("gp_price_AAPL loss is :", error_fusion)


