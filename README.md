# MT_github
![Image text](https://github.com/ZhangArcher/MT_github/blob/master/Process.png)




To do list(done):

0.data collection

1.stock price data handler,which can load stock price

2.simple Gaussian process fitting of stock prices based on different time scale

3.simple Gaussian process fusion for stock prices bewteen long term and short term

4.portfolios generation(portfolio data set generation)  based on standard / customized expected return ,risk model and objected function.

5.simple Gaussian Proccess fitting of single asset allocation based on the monthly prices. (just use simple kernel)

6.Simple Gaussian Proccess fusion of single asset allocation between longterm and short term

to do list(unfinish ,):

7. utilize the relationship among multi-Gaussian Process output (still being in test)


8. Using Gaussian Process fusion from MuDaFuGP.




details:

0.data collection:


to collect sotck prices data ,using package yfinance , according to different time scale , for example : daily prices, monthly prices , yearly prices


1.stock price data handler:


data_handler_price can load stock price and can return a dataframe of stock price based on a speical time interval and a customerzied timescale.



2.Simple Gaussian process fitting of stock prices based on different time scale with different fitting_length and pred_length

 data_wrapper_price_gp can load stock prices by data_handler.
 Using these stock prices , data_wrapper_gp can fit a Gaussian Process GP.
 Using GP , data_wrapper_gp can predict the future price.

    Term:
        fitting_length: the data size used to fit a Gaussian Process
        pred_length: prediction length,the length of the future time what we to predict
        excess_time:  the difference between long-term time point and short-term time point.

    Example (in formulas):
        Let us assumpt :
        There are 20 trading days each month and 252 trading days each year.

        Data=The data set about the daily prices between 01.06.2000 and 01.06.2002  ( 2 years)
        fitting_length = the size of Data = 252*2 = 504


        Gaussian_Process_fit(Data)=GP


        X:= next day


        GP.predict(X) = the predicted next day stock price  (the predicted daily stock price on 02.01.2002)=pred_Price

        here , we treat the corresponding mean of GP as  the predicted price.

        Hence,
        pred_Price := the predicted daily stock price on 02.01.2002
        target_Price :=the daily stock price on 02.01.2002

        Error = abs(target_Price-pred_Price)

        In this section, we want to know whether or not using a larger fitting_length and a smaller pred_length can reduce the Error.


    Experiment1:
        goal : test the effect of different pred_length.

        data set:
            long_1month_term_MSFT.csv : the data set about MSFT stock monthly price from 1995-01-22 to  2015-01-22.

        Process:
            1.Using different pred_length(5,10,15,20,25,30) to predict the price.
            2.To compare the prices from different pred_length

        result:
            A very strange result. The predicted error and the predicted result doesn't change itself with the change of fitting_length.

            We need to check the Gaussian Process package, whether or not I have utilize the package correctly.
            One possible reason is we are using sklearn.gaussian_process.GaussianProcessRegressor.
            It allows prediction without prior fitting.
            we need to do our prior fitting.










3.simple Gaussian process fusion for stock prices bewteen the long-term and short-term
    data_fusion_price can predict long-term stock price using short-term stock price.
    (To fuse short-term stock price into long-term stock price)


        Example (in formulas):
        Let us assumpt :
        There are 20 trading days each month and 252 trading days each year.

        D_short=The data set about the daily prices between 01.01.2000 and 16.01.2002  ( 2 years + 16 days =there are 504+5  daily prices (509 data points), while assuming there are 5 trading days between 01.01.2002 and 16.01.2002 )
        D_long= The data set about the monthly prices between 01.01.2000 and 01.01.2002 (2 years +1 month= there are 25 monthly prices (25 data points))

        fitting_length_long = the size of D_long.
        fitting_length_short = the size of D_short.

        Gaussian_Process_fit(D_short)=GP_short_term
        Gaussian_Process_fit(D_long)=GP_long_term

        The difference between D1 and D2 is 15 days.
        Here, excess_time = 15 days

        X_short_term:= next day
        X_long_term:= next month

        GP_short_term.predict(X_short_term) = the predicted next day stock price  (the predicted daily stock price on 17.01.2002)=pred_Price_short
        GP_long_term.predict(X_long_term) = the predicted next month stock price (the predicted monthly stock price in 02.2002)=pred_Price_long

        pred_Price_short:=the predicted next day stock price  (the predicted daily stock price on 17.01.2002)
        pred_Price_long:=the predicted next month stock price (the predicted monthly stock price in 02.2002)

        Price_short:= the daily stock price on 17.01.2002
        Price_long:= the monthly stock price on 01.02.2002

        It seems that we are trying to find a mapping function which can map pred_Price_short into Price_long.

        important note:
        here is the reason why don't we use GP_short_term.predict to predict the daily stock prices from 17.01.2002 to 01.02.2002.
        Because the predicted daily price will go to close to zero form the 3rd day(19.01.2002), while the prediction length is larger than 3.
        Hence , we have to shorten the prediction length.






3.1  simple Gaussian process fusion with the correction term
     we try to predict the long-term price using short-term GP with a correction term and then compare them with the long-term GP.
     Here , we don't use any fixed fitting_length.
     The fitting_length is increasing with the distance between start_point and end_point.

            Example (in formulas):

                our assumption :
                                GP_short_term.predict(X_short_term) +correction_term =GP_long_term.predict(X_long_term)
                                GP_long_term.predict(X_long_term)=Price_long

                in real-life situation :
                                GP_short_term.predict(X_short_term) +correction_term + Error =GP_long_term.predict(X_long_term)
                                GP_long_term.predict(X_long_term)+Error =Price_long

                Hence : Price_long - GP_short_term.predict(X_short_term) -correction_term  =error_1
                        Price_long - GP_long_term.predict(X_long_term)=error_2

                our goal (one reasonable goal) :
                    to prove : abs(error_1) <abs(error_2)

                Here , correction_term is a customized term ,
                options of correction_term includes :
                    1.the mean of the historical errors
                    2. a constant
                    ....

                our correction_term is the mean of the historical errors.








            Experiment2:
                We try to prove : abs(error_1) <abs(error_2) , while using a increasing fitting-length.



                data set:
                    long_1month_term_MSFT.csv : the data set about MSFT stock monthly price from 1995-01-22 to  2015-01-22.
                    short_1day_term_MSFT.csv : the data set about MSFT stock daily price from 1995-01-22 to  2015-01-22.




                Process:
                    start_time="2008-01-01 00:00:00"
                    end_time="2015-01-01 00:00:00"
                    we predict the MSFT monthly price from 2008-01-01 to 2015-01-01 using simple Gaussian process fusion with the correction term.

                    1.predict MSFT monthly price on 2008-02-01 using  data set 1 .
                    data set 1 is from  2008-01-01 to 2008-01-31 .
                    2.predict MSFT monthly price on 2008-03-01 using  data set 2 .
                    data set 2 is from  2008-01-01 to 2008-02-29.
                    3.predict MSFT monthly price on 2008-04-01 using  data set 3.
                    data set 3 is from  2008-01-01 to 2008-03-31.
                    ...........
                    n. predict MSFT monthly price on 2008-04-01 using  data set n.
                    data set n is from  2008-01-01 to 2014-12-31.


                    we need to test whether or not abs(error_1) <abs(error_2) while using the correction term.
                    correction_term=mean(historical error_1)

                result:

                    np.mean(abs(error_1))=0.00213301
                    np.mean(abs(error_2))=0.055280711
                    abs(error_1) <abs(error_2)
                    According to small data set test, abs(error_1) is really smaller than abs(error_2)

                    see :
                    test_result/Experiment2_pred_price_long_term.csv
                    test_result/Experiment2_pred_price_error_fusion_with_correction.csv






3.2 simple Gaussian process fusion without any correction term
      we try to predict the long-term price directly using short-term GP  and then compare them with the long-term GP.
            Example (in formulas):

                our assumption :
                                GP_short_term.predict(X_short_term)  =GP_long_term.predict(X_long_term)
                                GP_long_term.predict(X_long_term)=Price_long

                in real-life situation :
                                GP_short_term.predict(X_short_term)  + Error =GP_long_term.predict(X_long_term)
                                GP_long_term.predict(X_long_term)+Error =Price_long

                hence : Price_long - GP_short_term.predict(X_short_term)   =error_1
                        Price_long - GP_long_term.predict(X_long_term)= error_2

                our goal (one reasonable goal) :
                    we want to know the relationship between abs(error_1) and abs(error_2).


          Experiment3:
                we want to know the relationship between abs(error_1) and abs(error_2).

                data set:
                    long_1month_term_MSFT.csv : the data set about MSFT stock monthly price from 1995-01-22 to  2015-01-22.
                    short_1day_term_MSFT.csv : the data set about MSFT stock daily price from 1995-01-22 to  2015-01-22.




                Process:
                    start_time="2008-01-01 00:00:00"
                    end_time="2015-01-01 00:00:00"
                    we predict the MSFT monthly price from 2008-01-01 to 2015-01-01 using simple Gaussian process fusion with the correction term.

                    1.predict MSFT monthly price on 2008-02-01 using  data set 1 .
                    data set 1 is from  2008-01-01 to 2008-01-31 .
                    2.predict MSFT monthly price on 2008-03-01 using  data set 2 .
                    data set 2 is from  2008-01-01 to 2008-02-29.
                    3.predict MSFT monthly price on 2008-04-01 using  data set 3.
                    data set 3 is from  2008-01-01 to 2008-03-31.
                    ...........
                    n. predict MSFT monthly price on 2008-04-01 using  data set n.
                    data set n is from  2008-01-01 to 2014-12-31.




                result:
                    According to our small data set test, abs(error_1) is really smaller than abs(error_2)
                    np.mean(abs(error_1))=0.002055978
                    np.mean(abs(error_2))=0.055280711


                    see :
                    test_result/Experiment3_pred_price_error_fusion_without_correction.csv
                    test_result/Experiment3_pred_price_long_term.csv


    summary: Using non correction term maybe better than using a correction term.

             Experiment3A:
                we want to evaluate the parameter excess_time.
                (The rest of the setting is same to Experiment3 . Hence , there is not any correction term)
                data set:
                    long_1month_term_MSFT.csv : the data set about MSFT stock monthly price from 1995-01-22 to  2015-01-22.
                    short_1day_term_MSFT.csv : the data set about MSFT stock daily price from 1995-01-22 to  2015-01-22.

                Process:
                    start_time="2008-01-01 00:00:00"
                    end_time="2010-01-01 00:00:00"
                    we predict the MSFT monthly price from 2008-01-01 to 2015-01-01 using simple Gaussian process fusion without any correction term.

                    we predict the monthly price  with a increasing fitting length like last experiment but using different excess_time.

                    excess_time=[0,1,2,3,4,5,6,7,8,9]

                    Price_long - GP_short_term.predict(X_short_term)   =error_1
                    Price_long - GP_long_term.predict(X_long_term)= error_2



                result:
                excess_time=[0,1,2,3,4,5,6,7,8,9]
                average error_1:	0.063465226	, 0.001713583	,0.001713583	,0.001713583	,0.001713583	,0.001713583	,0.001713583	,0.001713583	,0.001713583	,0.001713583
                average error_2:    0.063920284	,0.063920284	,0.063920284	,0.063920284	,0.063920284	,0.063920284	,0.063920284	,0.063920284	,0.063920284	,0.063920284

                Hence, average  error_1 < average error_2.
                But very starnge. Why a new excess_time ccan not change the error?
                One possible reason that fitting_length is very large .
                Hence, adding new element by adding excess_time can not change the result.


                    see :
                    test_result/Experiment3A_price_fusion_different_excess_time.csv
                    test_result/Experiment3A_price_fusion_different_excess_time.xlsx





4.The portfolio generation(portfolio data set generation)  based on standard / customized expected return ,risk model and objected function.
    The portfolio generation is based on PyPortfolioOpt.
    (https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html)
    The portfolios generation is focusing on mean-variance optimization (MVO).
    MVO is going to calculate/optimize a portfolio according the expected returns and the risk model (covariance matrix).
    Using PyPortfolioOpt , we can easier generate the expected returns and the covariance matrix from the historical prices.
    And then we can generate a portfolio from the expected returns and the covariance matrix by PyPortfolioOpt.
    we need to choice how to calculate/generate the expected returns , the covariance matrix and portfolio.

    option:
        calculate the expected returns :
            1.mean_historical_return   : calculate the mean of the historical returns
            2.ema_historical_return
            3.capm_return
            4......


        calculate the covariance matrix :
            1.semicovariance
            2.CovarianceShrinkage.Ledoit Wolf shrinkage
            3.sample covariance
            4.....

        calculate the portfolio:
            1.EfficientSemivariance
            2.Efficient CVaR
            3.EfficientCDaR

        objective_functions:
             L2 regularisation
             ....

        (using objective_functions , the generated portfolios can be more stable. More stable means that the position of asset doesn't not go down into 0 suddenly)

        https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html#efficient-semivariance


    Our current selection is :
         the expected returns: mean_historical_return
         the covariance matrix :  semicovariance
         the portfolio:   EfficientSemivariance
         the objective_functions: L2 regularisation


    Data Generation:
    long-term portfolios:
        input :
            Monthly price Data of 34 different stocks  from 2001-01-19 to 2013-01-03 ( daily price and monthly price)
        output:
             long-term portfolios (from 2001-01-19 to 2013-01-03)  size:134

    short-term portfolios:
        input :
            Daily price Data of 34 different stocks  from 2001-01-19 to 2013-01-03 ( daily price and monthly price)
        output:
            short-term portfolios (from 2001-01-19 to 2013-01-03)  size:3006

    important note:
        At least now, why don't we use S&P 50 . Because they have different startup years.
        Some stock price data begin from 01-01-1990.
        But some maybe begin from 01-01-2006.
        Hence, we just select  the stocks whose the startup year is enough early.



5.Simple Gaussian Proccess fitting of asset allocation based on the monthly prices / daily price.
    we want to :
            to fit Gaussian Process for each asset using their historical position
            to predict the corresponding position(allocation) next time point.
            to prove that the predicted position is better than the original position


             For example  (in formulas):
                we are going to optimize the allocation of stock a,b,c.
                Portfolio:   Pi=(pi_a,pi_b,pi_c)
                while :
                 i is timestamp,
                 pi_a is the position of asset a at time point i.
                 pi_b is the position of asset b at time point i.
                 pi_c is the position of asset c at time point i.

                Let us assumpt : i=1,2,3,4
                Data_a=(p1_a,p2_a,p3_a)
                Data_b=(p1_b,p2_b,p3_b)
                Data_c=(p1_c,p2_c,p3_c)

                GP_a=Gaussian_Process_fit(Data_a)
                GP_b=Gaussian_Process_fit(Data_b)
                GP_a=Gaussian_Process_fit(Data_c)

                GP_a.predict(i=4)=pred_p4_a
                GP_b.predict(i=4)=pred_p4_b
                GP_c.predict(i=4)=pred_p4_c

                P4=(p4_a,p4_b,p4_b)
                pred_P4=(pred_p4_a,pred_p4_b,pred_p4_b)

                calculate_profit_by_portfolio(P4)=profit
                calculate_profit_by_portfolio(pred_P4)=pred_profit

                loss(score)=profit-pred_profit

                The ideal experiment result is: loss>0
                loss>0  means that our predicted portfolio is better than the original portfolio

                The real-life  experiment result  is :
                sometimes :loss>0
                sometimes :loss<0

    Note:  In our research , we just focus on long position. hence , pi_x >=0 .
           But pred_pi_x maybe smaller than 0 .
           Hence , we need to set all negative pred_pi_x into 0 .
           p4_a+p4_b+p4_b must be equal to 1
           But pred_p4_a+pred_p4_b+pred_p4_b  always not be 1 .
           Hence , we need to normalize (pred_p4_a,pred_p4_b,pred_p4_b).



     Experiment4:
        data set:
            long-term portfolios of 34 stocks (from 2001-01-19 to 2013-01-03)  size:134

        process:

            start=01.02.2006
            end=01-01-2011
            P1 is the long-term portfolios in 01.02.2006
            P2 is the long-term portfolios in 01.02.2006
            ....
            Pn is the long-term portfolios in 01-01-2011

            we need to build up independent GPs for each assets. (totally 34 GPs)

            The Gaussian process fitting use non-fixed fitting-length (the increasing fitting-length)
            The pred_Pi is generated by the predictions result from 34 GPs.
            And then we need to compute the profit ,pred_profit and loss.

            profit : how much money does the portfolio can earn? (in %)
            pred_profit : how much money does the predicted portfolio can earn? (in %)
            loss=pred_profit-profit

        result:

                average profit: 0.006852366
                average pred_profit: 0.001048108
                average loss: -0.005804258

        see: test_result/Experiment4_long_term_GP_loss.csv


     Experiment5:
            data set:
                short-term portfolios of 34 stocks (from 2001-01-19 to 2013-01-03)       size:3006

            process:

                start=01.02.2006
                end=2006-04-18
                P1 is the short-term portfolios in 01.02.2006
                P2 is the short-term portfolios in 02.02.2006
                ....
                Pn is the short-term portfolios in 01-04-2006

                we build up independent GPs for each assets. (totally 34 GPs)

                The Gaussian process fitting use non-fixed fitting length (the increasing length)
                The pred_Pi is composed of the prediction results from 34 GPs.

                And then we compute the profit ,pred_profit and loss.

                profit : how much money does the portfolio Pi can earn? (in %)

                pred_profit : how much money does the predicted portfolio pred_Pi can earn? (in %)

                loss=pred_profit-profit

            result:
                average profit: 0.000557206
                average pred_profit: 0.000284955
                average loss: -0.000272251

            see:test_result/Experiment5_short_term_GP_loss.csv

    Summarily, The simple GP prediction can just offer a negative effect.


6. Simple Gaussian Proccess fusion of  asset allocation between long-term and short-term

     can predict long-term stock price using short-term stock price.
    (To fuse short-term stock price into long-term stock price)

            For example (in formulas):

            our assumption :
                            GP_short_term.predict(X_short_term) +correction_term =GP_long_term.predict(X_long_term)
                            GP_long_term.predict(X_long_term)=Price_long

            in real-life situation :
                            GP_short_term.predict(X_short_term) +correction_term + Error =GP_long_term.predict(X_long_term)
                            GP_long_term.predict(X_long_term)+Error =Price_long

            hence : Price_long - GP_short_term.predict(X_short_term) -correction_term  =error_1
                    Price_long - GP_long_term.predict(X_long_term)=error_2

            our goal (one reasonable goal) :
                to proof : abs(error_1) <abs(error_2)

            Here , correction_term is a customized term ,
            options of correction_term includes :
                            1.the mean of the historical errors
                            2. a constant



            Experiment6:
            To test the performance of simple Gaussian proccess fusion of  asset allocation between long-term and short-term , while correction_term=0
            Hence, we don't use any correction_term
            data set:
                short-term portfolios of 34 stocks (from 2001-01-19 to 2013-01-03)  size:3006
                long-term portfolios of 34 stocks (from 2001-01-19 to 2013-01-03)  size:134
            process:

                start=01.01.2006
                end=01.01.2009
                excess_time=15

                P1 is the short-term portfolios in 01.02.2006
                P2 is the long-term portfolios in 02.02.2006
                ....
                Pn is the long-term portfolios in 01.01.2009

                we build up independent GPs for each assets. (totally 34 GPs)

                The Gaussian process fitting use non-fixed fitting-length (the increasing fitting-length)
                The pred_Pi are generated by the predictions result from 34 GPs.

                And then we can compute the profit ,pred_profit and loss.

                profit : how much money does the portfolio can earn? (in %)
                pred_profit : how much money does the long-term predicted portfolio can earn? (in %)
                fusion_profit: how much money does the fusion predicted portfolio can earn? (in %)

                fusion_loss=fusion_profit - profit
                pred_loss=pred_profit - profit

            result:
                average fusion_loss=-0.002383809
                average pred_loss=-0.004700145

                average fusion_loss <  average pred_loss
                As we can see , simple fusion is better than the directly GP prediction.



            see:
                Experiment6_portfolio_fusion_result.csv

    Hence , we can say ,simple fusion can reduce the error of GP prediction.


