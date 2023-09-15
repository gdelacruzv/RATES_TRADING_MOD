getwd()
setwd('C:/Users/gilberto/Documents/Econometrics')
DATA<- read.csv("RATE_DATA.csv")

library(forecast)
library(xts)
library(stats)
library(ggplot2)
library(opera)


# Create the models for the spot data 
# 


models = list() 

models$exp_inf = lm( EXPINF_10YR ~  SPREAD +	CPI_CHG +	FFR, DATA) 
# Get a summary of the model
model_summary <- summary(models$exp_inf)
# Display the coefficients
print(model_summary$coefficients[, 1])

# Display the associated p-values
print(model_summary$coefficients[, 4])

models$REAL_10T = lm( Real_10yr	~ AAA_Yield_R  +	BAA_Yield, DATA) 
# ####
#Momentum@
DATA$mom_Q=as.factor(ifelse(DATA$mom3 > 0, "Pos", "Neg")) 



# 
# Calculate residuals for the spot model 
# 
resids = list() 

resids$exp_infl = residuals(models$exp_inf) 

resids$REAL_10T = residuals(models$REAL_10T) 

resids$GAP_MODEL_RESIDS = resids$REAL_10T + resids$exp_infl

# Calculate forward deltas#
DATA$UST_10YR.DELTA_Q = DATA$Delta_q

# Define objective functions 
# 
MIN_Z = 0.35
z.score <- function(x) (x - mean(x)) / sd(x, na.rm=T)
print(DATA$UST_10YR.DELTA_Q)



DATA$UST_10YR_1FWDz = z.score(DATA$UST_10YR.DELTA_Q)

object = data.frame( 
  UST_10YR_1FWD.buy = z.score(DATA$UST_10YR_1FWDz) > MIN_Z,
  UST_10YR_1FWD.sell = z.score(DATA$UST_10YR_1FWDz) < -(MIN_Z)
)
object$UST_10YR_1FWD.buy 
# Create models for the forward trading objectives 
# 
# Minimal models for swap spreads 
# 
models$gap_model.T10.buy.min = glm( object$UST_10YR_1FWD.buy ~ 
                                      resids$GAP_MODEL_RESIDS +mom3, 
                                family=binomial(), 
                                data=DATA) 


models$gap_model.T10.sell.min = glm(object$UST_10YR_1FWD.sell ~ 
                                   resids$GAP_MODEL_RESIDS + mom3, 
                                 family=binomial(), 
                                 data=DATA) 

# 
# Apply trading models to data, giving P&L 
# 
# Several experiments showed that 0.5 here is a good 
# balance of profit, drawdown, and sharpe ratio 
# 
PROB_THRESH = 0.5 
gen.signal <- function(model) { 
  pred = predict(model) 
  odds = exp(pred) 
  prob = odds / (1 + odds) 
  sig = prob > PROB_THRESH 
  return(sig) 
} 
calc.pl <- function(model, fwd) {
  pl = ifelse(gen.signal(model), fwd, 0) 
  return(pl) 
} 


# Generate trade signals, buy and sell, for all models 
# 
signal = list() 
signal$ss10.fwd10.buy.min = gen.signal(models$gap_model.T10.buy.min) 
signal$ss10.fwd10.sell.min = gen.signal(models$gap_model.T10.sell.min) 
signal$ss10.fwd10.buy.min 

# Generate actual outcomes 
# 
outcomes = list() 
outcomes$ss10 = ifelse(DATA$UST_10YR.DELTA_Q > 0, "Up", "Dn") 
# Generate 2x2 contingency tables for all models 
# 
tables = list() 
make.table <- function(sig) { 
  tbl = table(signal = sig, 
              outcome = outcomes$ss10, 
              exclude = NA) 
  return(tbl) 
} 
tables$ss10.fwd10.buy.min = make.table(signal$ss10.fwd10.buy.min) 
tables$ss10.fwd10.sell.min = make.table(signal$ss10.fwd10.sell.min) 
rm(make.table) 


# Functions to calculate odds ratios 
# 
odds.ratio <- function(tbl) { 
  return ((tbl[1,1] * tbl[2,2]) / (tbl[1,2] * tbl[2,1])) 
} 
odds.ratio.ase <- function(tbl) sqrt(1/tbl[1,1] + 1/tbl[1,2] + 1/tbl[2,1] + 
                                       1/tbl[2,2]) 
odds.ratio.ci <- function(tbl) { 
  logOR = log(odds.ratio(tbl)) 
  ase = odds.ratio.ase(tbl) 
  return (c(exp(logOR - 1.96*ase), exp(logOR + 1.96*ase))) 
} 
relative.risk <- function(tbl) { 
  return (tbl[1,1] / (tbl[1,1] + tbl[1,2])) / (tbl[2,1] / (tbl[2,1] + 
                                                             tbl[2,2])) 
}


# Calculate P&L, buy and sell and combined, for all models 
# 
pl = list() 
pl$ss10.fwd10.buy.min = calc.pl(models$gap_model.T10.buy.min, 
                                DATA$Delta_1) 
pl$ss10.fwd10.sell.min = calc.pl(models$gap_model.T10.sell.min,-
                                      (DATA$Delta_1)) 
pl$ss10.fwd10.min = pl$ss10.fwd10.buy.min + pl$ss10.fwd10.sell.min 

rm(calc.pl) 

# Plot the cumulative P&L 
# 
par(mfrow=c(1,1)) 
plot(cumsum(pl$ss10.fwd10.min), typ='l', ylab='P&L: Final model (x$1,000)', 
     xlab='Day') 

summary.pl <- function(v, buyTbl, sellTbl) { 
  cat("\tP&L:", 
      sum(v), 
      "total =", 
      sum(v)/(length(v)/12), "p.a.", 
      "\n") 
  cat("\tAnn Sharpe ratio:", 
      12 * mean(v) / (sqrt(12) * sd(v)), 
      "\n"); 
  cat("\tMax drawdown:", max(cummax(cumsum(v)) - cumsum(v)), "\n"); 
   
} 

cat("Minimal model P&L:\n"); 
summary.pl(pl$ss10.fwd10.min, tables$ss10.fwd10.buy.min, 
           tables$ss10.fwd10.sell.min)
rm(summary.pl)


