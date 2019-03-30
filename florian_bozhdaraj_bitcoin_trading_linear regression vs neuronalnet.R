# Topic: regression vs. simple feedforward based on nn
#   Application to bitcoin
# First week

rm(list=ls())

library(neuralnet)
library(boot)
library(plyr) 
library(MASS)
library(fGarch)
library(xts)
library(readr)


data.set <- read_delim("C:/Users/Florian/Desktop/datasets/db_export_table_price_2019-03-28.csv", 
                       ";", escape_double = FALSE, trim_ws = TRUE)

data.set <- as.data.frame(data.set[,1:2])
data.set <- na.omit(data.set)
dat <- as.xts(as.numeric(data.set$bitcoin), order.by = as.Date(data.set$Date))
head(dat)



x<-na.omit(diff(log(dat)))
head(x)


data_mat<-cbind(x,lag(x),lag(x,k=2),lag(x,k=3),lag(x,k=4),lag(x,k=5),lag(x,k=6))
head(data_mat)

# Check length of time series before na.exclude
dim(data_mat)
data_mat<-na.exclude(data_mat)
# Check length of time series after removal of NAs
dim(data_mat)
head(data_mat)
tail(data_mat)


in_out_sample_separator<-"2019-03-20"

target_in<-data_mat[paste("/",in_out_sample_separator,sep=""),1]
tail(target_in)
explanatory_in<-data_mat[paste("/",in_out_sample_separator,sep=""),2:ncol(data_mat)]
tail(explanatory_in)

target_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),1]
head(target_out)
tail(target_out)
explanatory_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),2:ncol(data_mat)]
head(target_out)
tail(explanatory_out)

train<-cbind(target_in,explanatory_in)
test<-cbind(target_out,explanatory_out)
head(test)
tail(test)
nrow(test)


# Fitting linear model to log-returns (could be applied to standardized log-returns: returns divided by vola)
lm.fit <- lm(target_in~explanatory_in)#mean(target_in)plot(cumsum(target_in))plot(log(dat$Bid))
summary(lm.fit)

# Without intercept
lm.fit <- lm(target_in~explanatory_in-1)
summary(lm.fit)


#   Without intercept
predicted_lm<-explanatory_out%*%lm.fit$coef

# With intercept: we ahve to add a column of 1s to the explanatory data (for the additional intercept)
if (length(lm.fit$coef)>ncol(explanatory_out))
  predicted_lm<-cbind(rep(1,nrow(explanatory_out)),explanatory_out)%*%lm.fit$coef



# Test MSE: in-sample vs. out-of-sample
MSE.in.lm<-mean(lm.fit$residuals^2)
MSE.out.lm <- sum((predicted_lm - target_out)^2)/nrow(test)
c(MSE.in.lm,MSE.out.lm)


# 3.i Trading performance
perf_lm<-(sign(predicted_lm))*target_out


sharpe_lm<-sqrt(365)*mean(perf_lm,na.rm=T)/sqrt(var(perf_lm,na.rm=T))

par(mfrow = c(2,1))
plot(cumsum(perf_lm),main=paste("Linear regression cumulated performances out-of-sample, sharpe=",round(sharpe_lm,2),sep=""))


# 4. Neural net fitting
head(data_mat)


# Scaling data for the NN
maxs <- apply(data_mat, 2, max) 
mins <- apply(data_mat, 2, min)
# Transform data into [0,1]  
scaled <- scale(data_mat, center = mins, scale = maxs - mins)
head(scaled)


apply(scaled,2,min)
apply(scaled,2,max)


# Train-test split
train_set <- scaled[paste("/",in_out_sample_separator,sep=""),]
test_set <- scaled[paste(in_out_sample_separator,"/",sep=""),]

train_set<-as.matrix(train_set)
test_set<-as.matrix(test_set)



colnames(train_set)<-paste("lag",0:(ncol(train_set)-1),sep="")
n <- colnames(train_set)
# Model: target is current bitcoin, all other variables are explanatory  
f <- as.formula(paste("lag0 ~", paste(n[!n %in% "lag0"], collapse = " + ")))

# Set/fix the random seed 
set.seed(1)
nn <- neuralnet(f,data=train_set,hidden=c(6,4,2),linear.output=F)   #hidden = (Anzahl Layer, Anzahl Neuronen)
#plot(nn)



# In sample performance
MSE.in.nn<-mean(((train_set[,1]-nn$net.result[[1]])*(max(data_mat[,1])-min(data_mat[,1])))^2) # rücktransformation

# Out-of-sample performance
pr.nn <- compute(nn,test_set[,2:ncol(test_set)])

predicted_scaled<-pr.nn$net.result



# Results from NN are normalized (scaled)
# Descaling for comparison
predicted_nn <- predicted_scaled*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
test.r <- test_set[,1]*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
# Calculating MSE
MSE.out.nn <- sum((test.r - predicted_nn)^2)/nrow(test_set)

# Compare in-sample and out-of-sample
c(MSE.in.nn,MSE.out.nn)

# Compare Regression and nn in-sample
print(paste(MSE.in.lm,MSE.in.nn))

# Compare Regression and nn out-of-sample
print(paste(MSE.out.lm,MSE.out.nn))


# 4.f Trading performance
perf_nn<-(sign(predicted_nn))*target_out


sharpe_nn<-sqrt(365)*mean(perf_nn,na.rm=T)/sqrt(var(perf_nn,na.rm=T))

plot(cumsum(perf_nn),main=paste("NN cumulated performances out-of-sample, sharpe=",round(sharpe_nn,2),sep=""))

