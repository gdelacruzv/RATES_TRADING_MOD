# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 01:18:04 2023

@author: Gilberto
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
%matplotlib inline
# Set directory and read data
data = pd.read_csv('C:/Users/gilberto/Documents/Econometrics/RATE_DATA_v2.csv')

# Convert 'Code' column to datetime format for splitting
data['Code'] = pd.to_datetime(data['Code'])

# Splitting the dataset
train_data = data[data['Code'].dt.year < 2019]
test_data = data[data['Code'].dt.year >= 2019]


# Create the models for the spot data
#exp_inf_model = smf.ols(formula='EXPINF_10YR ~ SPREAD + CPI_CHG + FFR', data=data).fit()
#print(exp_inf_model.params)
#print(exp_inf_model.pvalues)

#real_10t_model = smf.ols(formula='Real_10yr ~ AAA_Yield_R + BAA_Yield', data=data).fit()
#print(real_10t_model.summary())

ust_10t_model = smf.ols(formula='UST_10YR ~ EXPINF_10YR + Real_10yr', data=train_data).fit()
# Calculate residuals for the training and testing datasets
train_data['UST_10YR_resid'] = ust_10t_model.resid
test_data['UST_10YR_resid'] = test_data['UST_10YR'] - ust_10t_model.predict(test_data)
# ... (and so on for other residuals)

ust_10t_model = smf.ols(formula='UST_10YR ~ EXPINF_10YR + Real_10yr', data=test_data).fit()
print(ust_10t_model.summary())
data['mom_Q'] = np.where(data['mom3'] > 0, "Pos", "Neg")

# Calculate residuals for the spot model
#data['exp_infl_resid'] = exp_inf_model.resid
#data['REAL_10T_resid'] = real_10t_model.resid
#data['GAP_MODEL_RESIDS'] = data['REAL_10T_resid'] + data['exp_infl_resid']
data['ust_10T_resid'] = ust_10t_model.resid
MIN_Z = 0.45

z_score = lambda x: (x - np.mean(x)) / np.std(x, ddof=1)  # ddof=0 for population std deviation
train_data['UST_10YR_1FWDz'] = z_score(train_data['delta_1'])

train_data['UST_10YR_1FWD_buy'] = train_data['UST_10YR_1FWDz'] > MIN_Z
train_data['UST_10YR_1FWD_sell'] = train_data['UST_10YR_1FWDz'] < -MIN_Z


#gap_model_buy = smf.glm('UST_10YR_1FWD_buy ~ ust_10T_resid + FFR + BAA_Yield', data=test_data, family=sm.families.Binomial()).fit()
#print(gap_model_buy.summary())
#gap_model_sell = smf.glm('UST_10YR_1FWD_sell ~  ust_10T_resid + FFR + BAA_Yield', data=test_data, family=sm.families.Binomial()).fit()
#print(gap_model_sell.summary())

gap_model_buy = smf.glm('UST_10YR_1FWD_buy ~ UST_10YR_resid + FFR + BAA_Yield', data=train_data, family=sm.families.Binomial()).fit()
gap_model_sell = smf.glm('UST_10YR_1FWD_sell ~  UST_10YR_resid + FFR + BAA_Yield', data=train_data, family=sm.families.Binomial()).fit()







# Adjust outcomes calculation for test data
test_data['outcomes'] = np.where(test_data['delta_1'] > 0, "Up", "Dn")


# Adjust outcomes calculation for test data
train_data['outcomes'] = np.where(train_data['delta_1'] > 0, "Up", "Dn")




prob = gap_model_buy.predict()
prob
prob1 = gap_model_sell.predict()
prob1

def gen_signal(model, df_to_predict, threshold=0.70):
    prob = model.predict(df_to_predict)
    return prob > threshold

# Usage example:
PROB_THRESH_BUY = 0.65
PROB_THRESH_SELL = 0.70

train_data['Signal_Buy'] = gen_signal(gap_model_buy, train_data, PROB_THRESH_BUY)
train_data['Signal_Sell'] = gen_signal(gap_model_sell, train_data, PROB_THRESH_SELL)

test_data['Signal_Buy'] = gen_signal(gap_model_buy, test_data, PROB_THRESH_BUY)
test_data['Signal_Sell'] = gen_signal(gap_model_sell, test_data, PROB_THRESH_SELL)

train_data['outcomes'] = np.where(train_data['delta_1'] > 0, "Up", "Dn")

train_table_buy = pd.crosstab(train_data['Signal_Buy'] , train_data['outcomes'], rownames=['signal'], colnames=['outcome'])
train_table_sell = pd.crosstab(train_data['Signal_Sell'], train_data['outcomes'], rownames=['signal'], colnames=['outcome'])

test_table_buy = pd.crosstab(test_data['Signal_Buy'] , test_data['outcomes'], rownames=['signal'], colnames=['outcome'])
test_table_sell = pd.crosstab(test_data['Signal_Sell'], test_data['outcomes'], rownames=['signal'], colnames=['outcome'])

# For Train Data
train_buy_accuracy = train_table_buy.loc[True, "Up"] / train_table_buy.loc[True].sum() if True in train_table_buy.index and "Up" in train_table_buy.columns else 0
train_sell_accuracy = train_table_sell.loc[True, "Dn"] / train_table_sell.loc[True].sum() if True in train_table_sell.index and "Dn" in train_table_sell.columns else 0

# For Test Data
test_buy_accuracy = test_table_buy.loc[True, "Up"] / test_table_buy.loc[True].sum() if True in test_table_buy.index and "Up" in test_table_buy.columns else 0
test_sell_accuracy = test_table_sell.loc[True, "Dn"] / test_table_sell.loc[True].sum() if True in test_table_sell.index and "Dn" in test_table_sell.columns else 0

print("Train Buy Signal Accuracy: {:.2%}".format(train_buy_accuracy))
print("Train Sell Signal Accuracy: {:.2%}".format(train_sell_accuracy))
print("Test Buy Signal Accuracy: {:.2%}".format(test_buy_accuracy))
print("Test Sell Signal Accuracy: {:.2%}".format(test_sell_accuracy))


def calc_pl_buy(signal, outcomes, fwd):
    return np.where(signal & (outcomes == "Up"), fwd, 0)

def calc_pl_sell(signal, outcomes, fwd):
    return np.where(signal & (outcomes == "Dn"), -fwd, 0)  # Negative as it's for sell

def calc_combined_pl(data):
    pl_buy = calc_pl_buy(data['Signal_Buy'], data['outcomes'], data['delta_1'])
    pl_sell = calc_pl_sell(data['Signal_Sell'], data['outcomes'], data['delta_1'])
    return pl_buy + pl_sell
# Usage example:
pl_train_combined = calc_combined_pl(train_data)
# Usage example:
pl_test_combined = calc_combined_pl(test_data)

 
# Plotting

plt.xlabel('Day')
plt.ylabel('P&L: Final model (x$1,000)')
plt.plot(np.cumsum(pl_train_combined))
plt.plot(np.cumsum(pl_test_combined))
plt.show()

# Summary function
def summary_pl(v):
    total_pl = sum(v)
    ann_return = total_pl / (len(v) / 12)
    ann_sharpe = 12 * np.mean(v) / (np.sqrt(12) * np.std(v))
    max_drawdown = max(np.maximum.accumulate(v) - v)
    print(f"P&L: {total_pl} total = {ann_return} p.a.")
    print(f"Ann Sharpe ratio: {ann_sharpe}")
    print(f"Max drawdown: {max_drawdown}")

print("Minimal model P&L:")
summary_pl(pl_train_combined)
summary_pl(pl_test_combined)