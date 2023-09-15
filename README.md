# RATES_TRADING_MOD
Econometrics Analysis of Rate Data

The provided R script mainly carries out a financial econometric analysis of interest rate data. The script's functionalities can be segmented into different sections for clarity

1.Setting the Working Directory and Data Loading:
The working directory is established and data is loaded from a CSV file named "RATE_DATA.csv" located in the specified directory.

2.Library Imports:
Necessary libraries such as forecast, xts, stats, ggplot2, and opera are imported to support various data handling, modeling, and visualization tasks.


3.Linear regression models are fitted to the data:
        An inflation expectation (EXPINF_10YR) model using explanatory variables like spread, CPI change, and federal funds rate (FFR).
        A model for real 10-year rates (Real_10yr) using AAA and BAA yield rates as predictors.
        The coefficients and associated p-values of the inflation expectation model are printed for interpretation.

4.Momentum Indicator Creation:
        A momentum indicator (mom_Q) is created which categorizes momentum over a three-period range as either positive or negative.

5.Residual Calculation:
        Residuals are calculated for the fitted models, and a combined residual is also computed.

6.Objective Function Definition for Trading:
         z-score function is defined to standardize a series. 
         Trading objectives are defined using the z-scores of forward values, setting thresholds for buy and sell signals.

7.Modeling for Trading Objectives:
        Logistic regression models are created to predict buy/sell objectives using the combined residuals and momentum.

8.Signal Generation and Contingency Tables:
        A function to generate trading signals is defined based on predicted probabilities.
        Contingency tables are constructed to show the relationship between the generated signals and actual outcomes.

9.Odds Ratio and Relative Risk Calculation:
        Functions are defined to compute odds ratios, their confidence intervals, and relative risks from the contingency tables.

10.Profit & Loss (P&L) Computation:
        P&L is computed using the generated signals and actual forward values.
        The cumulative P&L is plotted over time.

11.P&L Summary:
         A summary of P&L, including total profit, annualized Sharpe ratio, and maximum drawdown, is printed for interpretation.
