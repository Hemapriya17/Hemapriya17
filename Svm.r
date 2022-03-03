library(quantmod)
library(lubridate)
library(e1071)
library(rpart)
library(rpart.plot)
library(ROCR) 
options(warn = -1) 
a <- c('AAPL', 'FB', 'GE', 'GOOG', 'GM', 'IBM', 'MSFT') 
for (i in 1:length(a)) 
{ 
SYM <- a[i] 
print('-------------------------------------------------------------------- -----') 
print(paste('Predicting the output for', SYM, sep = ' ')) 
trainPerc <- 0.75 
date <- as.Date(Sys.Date() - 1) 
endDate <- date#as.Date("2016-01-01") 
d <- as.POSIXlt(endDate) 
d$year <- d$year - 2 
startDate <- as.Date(d) 
STOCK <- getSymbols( SYM, env = NULL, src = "yahoo", from = startDate, to = endDate ) 
RSI3 <- RSI(Op(STOCK), n = 3) 
#Calculate a 3-period relative strength index (RSI) off the open price 
EMA5 <- EMA(Op(STOCK), n = 5) 
#Calculate a 5-period exponential moving average (EMA) 
EMAcross <- Op(STOCK) - EMA5 
#Let us explore the difference between the open price and our 5-period EMA 
MACD <- MACD(Op(STOCK), fast = 12, slow = 26, signal = 9) 
#Calculate a MACD with standard parameters 
MACDsignal <- MACD[, 2] 
#Grab just the signal line to use as our indicator. 
SMI <- SMI( Op(STOCK), n = 13, slow = 25, fast = 2, signal = 9 ) 
#Stochastic Oscillator with standard parameters 
SMI <- SMI[, 1] 
#Grab just the oscillator to use as our indicator 
WPR <- WPR(Cl(STOCK), n = 14) 
WPR <- WPR[, 1] 
ADX <- ADX(STOCK, n = 14) 
ADX <- ADX[, 1] 
CCI <- CCI(Cl(STOCK), n = 14) 
CCI <- CCI[, 1] 
CMO <- CMO(Cl(STOCK), n = 14) 
CMO <- CMO[, 1] 
ROC <- ROC(Cl(STOCK), n=2) 
ROC <- ROC[, 1] 
#DPO <- DPO(Cl(STOCK), n = 10) 
#DPO <- DPO[, 1] 
PriceChange <- Cl(STOCK) - Op(STOCK) 
#Calculate the difference between the close price and open price 
Class <- ifelse(PriceChange > 0, 'UP', 'DOWN') 
#Create a binary classification variable, the variable we are trying to predict. 
DataSet <- data.frame(Class, RSI3, EMAcross, MACDsignal, SMI, WPR, ADX, CCI, CMO, ROC) 
#Create our data set 
colnames(DataSet) <- c( "Class","RSI3", "EMAcross", "MACDsignal", "Stochastic", 
                                             "WPR", "ADX", "CCI", "CMO", "ROC" ) 
#Name the columns 
#DataSet <- DataSet[-c(1:33), ] 
#Get rid of the data where the indicators are being calculated 
TrainingSet <- DataSet[1:floor(nrow(DataSet) * trainPerc),] 
#Use 2/3 of the data to build the tree 
TestSet <- DataSet[(floor(nrow(DataSet) * trainPerc) + 1):nrow(DataSet),] 
#And leave out 1/3 data to test our strategy 
SVM <- svm(
Class ~ RSI3 + EMAcross + WPR + ADX + CMO + CCI + ROC, 
data = TrainingSet, 
kernel = "radial", 
type = "C-classification", 
na.action = na.omit, 
cost = 1, 
gamma = 1 / 5 ) 

print(SVM) 
confmat <- table(predict(SVM, TestSet, type = "class"), 
TestSet[, 1], 
dnn = list('predicted', 'actual')) 
print(confmat)
tryCatch({ 
acc <- (confmat[1, "DOWN"] + confmat[2, "UP"]) * 100 / (confmat[2, "DOWN"] + confmat[1, "UP"] 
             + confmat[1, "DOWN"] + confmat[2, "UP"]) 
#if (acc > 60) { 
xy <- paste('SVM : Considering the output for', SYM, sep = ' ') 
yz <- paste('Accuracy =', acc, sep = ' ') 
out <- paste(xy, yz, sep = '\n') 
print(out) 
write(out, file = "out", append = TRUE, sep = "\n\n") 
#} 
}, 
error = function(e) { 
}) 
predds <- data.frame(predict(SVM, TestSet), TestSet$Class) 
colnames(predds) <- c("pred", "truth") 
predds[, 1] <- ifelse(predds[, 1] == 'UP', 1, 0) 
predds[, 2] <- ifelse(predds[, 2] == 'UP', 1, 0) 
pred <- prediction(predds$pred, predds$truth) 
perf = performance(pred, measure = "tpr", x.measure = "fpr") 
auc.perf = performance(pred, measure = 'auc', col = "red") 
rmse.perf = performance(pred, measure = 'rmse') 
print(paste('RMSE =', rmse.perf@y.values), sep = ' ') 
print(paste('AUC =', auc.perf@y.values), sep = ' ') 
plot(perf, col = 1:10) 
abline(a = 0, b = 1, col = "red")
print('-------------------------------------------------------------------- -----') }


