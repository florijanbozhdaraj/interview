# Packages laden
{
  library(tseries)
  library(fGarch)
  library(dendroTools)
  library(data.table) 
}




# read data

library(readr)
library(xts)
#library(lubridate)
library(fGarch)
data.set <- read_delim("C:/Users/Florian/Desktop/datasets/db_export_table_price_2019-03-28.csv", 
                       ";", escape_double = FALSE, trim_ws = TRUE)



data.set <- as.data.frame(data.set[,1:2])
data.set <- na.omit(data.set)
dat <- as.xts(as.numeric(data.set$bitcoin), order.by = as.Date(data.set$Date))
head(dat)


# plot last, bid and ask in single figure names(dat)
par(mfrow=c(2,2))
plot(dat, main="Prices")
plot(log(dat), main="Log-prices")  #tail(dat$Bid)
plot(diff(log(dat)), main="Log-returns")
par(mfrow=c(1,1))


# how many days to forecast:
fors <- 5


# "Stationarity":
x<-dat[1:(length(dat)-fors)]
par(mfrow=c(2,1))
ts.plot(x)
# log-returns bilden
y<-diff(log(x))
y <- na.omit(y)
ts.plot(y, main = 'Diff-logs')

# ACF of log-returns und squared log-returns
par(mfrow = c(3,1))
acf(y) # Acf does not exist
acf(y^2) # at squared long acf, memory
acf(abs(y)) # for a long acf, memory

ts.plot(abs(y))  # large log returns tend to be followed by big ones, followed by small ones
ts.plot(y^2)
par(mfrow = c(1,1))



## Fit optimal ARMA~GARCH Modell:


# Step 1: Check significance of paramaters:
distri = 'norm'   # Start with normal distribution

# First Start with a Garch Model:
# Garch  (Symmetric Risk, no autocorrelation):
Model<-garchFit(~garch(1,1),data=y,delta=2,include.delta=F,include.mean=T, cond.dist = distri)             
summary(Model)
# mu     1.023e-03   6.641e-04    1.541    0.123    No significance for mu, therefore set include.mean=F


# Step 2: Check distribution 
Model<-garchFit(~garch(1,1),data=y,delta=2,include.delta=F,include.mean=F, cond.dist = distri)
summary(Model)
#                                Statistic    p-Value     
# Jarque-Bera Test   R    Chi^2  5554.983     0           # p-values >0.05 OK (fake, but change to t-distribution would not be bettr here)
# Shapiro-Wilk Test  R    W      0.9076031    0           # # p-values >0.05 OK (fake, but change to t-distribution would not be bettr here)


# Step 3: Check dependence structure
#                                Statistic p-Value     
# Ljung-Box Test     R    Q(10)  41.76181  8.266872e-06 
# Ljung-Box Test     R    Q(15)  45.59866  6.157237e-05
# Ljung-Box Test     R    Q(20)  55.37657  3.606455e-05

# Standardised Residuals Tests:
# All p-values < 0.05, therefore we add an arma model to the garch
# Checked it by starting with arma(0,1) to arma(0,6). MA(6)~Garch(1,1) meet the requirements of the p-values > 0.05.

# arma+garch  (Symmetric Risk with autocorrelation):
Model<-garchFit(~arma(0,6)+garch(1,1),data=y,delta=2,include.delta=F,include.mean=F, cond.dist = distri)   
summary(Model)


# Step 4: Check if garch model parameter enough:
# Standardised Residuals Tests:
#                               Statistic  p-Value     
# Ljung-Box Test     R^2  Q(10)  12.18855  0.2726355   
# Ljung-Box Test     R^2  Q(15)  15.24434  0.4339645   
# Ljung-Box Test     R^2  Q(20)  17.22824  0.6381072   

# All p-values >0.05, therefore garch(1,1) is enough




## redisiduals 
eps = Model@residuals # Residuals (not yet standardized) (= epsilon_t)

u = eps/Model@sigma.t # standardized residuals (= ut)
# Standardization by adding the
# Inverse conditional vola adjusts the conditional volatility.

par(mfrow=c(2,2))
acf(u)           
acf(u^2)          # if something exists in squared ACFs, then increase model order, otherwise switch to APARCH!
acf(eps)
acf(eps^2)
par(mfrow=c(1,1))
# MA(6)~GARCH(1,1) seems to be a good model!



## Forecasting (5 days):
predi <- predict(Model, fors)
sign(predi$meanForecast)  # Prediction
# -1: Decreasing tmrw: Sell today
# +1: Increasing tmrw: Buy Today
sign(as.numeric(y[(length(y)-fors+1):length(y)]))


# Vizualization:
# 4.f Trading performance
dat <- diff(dat)
target_out<-dat[(length(dat)-fors):length(dat)]


perf_nn<-(sign(predi$meanForecast))*target_out[-1]


sharpe_nn<-sqrt(365)*mean(perf_nn,na.rm=T)/sqrt(var(perf_nn,na.rm=T))

plot(cumsum(perf_nn),main=paste("GARCH cumulated performances out-of-sample, sharpe=",round(sharpe_nn,2),sep=""))




