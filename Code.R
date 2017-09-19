setwd("C:/Users/PraveenKotha/Desktop/loan problem")
#install.packages(c("e1071","caret","doSNOW","ipred","xgboost"))
#install.packages("Hmisc")
#install.packages("missForest")
#install.packages("e1071")

##loading required libraries
library(missForest)
library(Hmisc)
library(caret)
library(doSNOW)
library(ggplot2)
library(dplyr)
library(MASS)
library(rpart)
library(e1071)
library(rpart.plot)
library(caret)

##Reading data files
l_train<-read.csv("train.csv",na.strings=c("","NA"))
l_test<-read.csv("test.csv",na.strings=c("","NA"))

##Combining the data
l_test$Loan_Status<-'Y'
l_total<-rbind(l_train,l_test)

##Inspecting the data
str(l_train)
summary(l_train)

##Plots for understanding data
qplot(x=LoanAmount,data=loan_train,fill=Loan_Status)
qplot(x=Property_Area,data=loan_train,fill=Loan_Status)
qplot(x=Credit_History,data=loan_train,fill=Loan_Status)
qplot(x=Loan_Amount_Term,data=loan_train,fill=Loan_Status)
qplot(x=ApplicantIncome,data=loan_train,fill=Loan_Status)
qplot(x=Self_Employed,data=loan_train,fill=Loan_Status)
qplot(x=Dependents,data=loan_train,fill=Loan_Status)
qplot(x=Married,data=loan_train,fill=Loan_Status)
qplot(x=Gender,data=loan_train,fill=Loan_Status)

##Cleaning the data
l_total$Credit_History<-as.factor(l_total$Credit_History)

l_total$Dependents<-chartr('+', ' ', l_total$Dependents)

l_total$Dependents<-as.factor(l_total$Dependents)

detach("package:dplyr
       ", character.only = TRUE)
library("dplyr", character.only = TRUE)

l_total<-select(l_total,-Loan_ID)

##Imputing missing values
l_total_imp <- missForest(l_total)
l_total<-l_total_imp$ximp


##Label encoding
l_total$Gender<-as.factor(ifelse(l_total$Gender=="Male",1,0))
l_total$Married<-as.factor(ifelse(l_total$Married=="Yes",1,0))
l_total$Education<-as.factor(ifelse(l_total$Education=="Graduate",1,0))
l_total$Self_Employed<-as.factor(ifelse(l_total$Self_Employed=="Yes",1,0))
l_total$Property_Area<-as.factor(ifelse(l_total$Property_Area=="Rural",1,
                                        ifelse(l_total$Property_Area=="Semiurban",2,3)))
l_total$CoApplicant<-ifelse(l_total$CoapplicantIncome==0,0,1)
l_total$CoApplicant<-as.factor(l_total$CoApplicant)

l_total$ApplicantIncome<-l_total$ApplicantIncome+l_total$CoapplicantIncome
l_total<-select(l_total,-CoapplicantIncome)

l_total$loan_pm<-l_total$LoanAmount/l_total$Loan_Amount_Term


l_total<-l_total%>%select(-Loan_Status,Loan_Status)

##One hot encoding
l_dummy <- dummyVars(~ ., data = l_total[,-ncol(l_total)])
lp.dummy <- predict(l_dummy, l_total[, -ncol(l_total)])
l_total<-as.data.frame(lp.dummy)


##Dividing data into training and test sets
newl_train<-l_total[1:nrow(l_train),]
newl_test<-l_total[-(1:nrow(l_train)),]
table(newl_train$ApplicantIncome)

newl_train$Loan_Status<-l_train$Loan_Status
newl_test$Loan_Status<-'Y'

##Removing Outliers
newl_train<-subset(newl_train,newl_train$LoanAmount<=600 & newl_train$LoanAmount>25)
newl_train<-subset(newl_train,newl_train$ApplicantIncome<50000)
newl_train<-subset(newl_train,newl_train$Loan_Amount_Term>70)



##Building Xgboost Machine learning model
library(caret)
library(doSNOW)

set.seed(123)
train.control3 <- trainControl(method = "repeatedcv",
                               number = 5,
                               repeats = 3,
                               search = "grid")

tune.grid3 <- expand.grid(eta = c(0.1,0.2,0.3),
                          nrounds = c(30, 40, 55),
                          max_depth = c(15,17,25),
                          min_child_weight = c(5,7,9,11),
                          colsample_bytree = c(0.6, 0.7, 0.8,0.9),
                          gamma = 1,
                          subsample = 1
)
View(tune.grid3)

cl3 <- makeCluster(3, type = "SOCK")


registerDoSNOW(cl3)

caret.cv3 <- train(Loan_Status ~ ., 
                   data = newl_train,
                   method = "xgbTree",
                   tuneGrid = tune.grid3,
                   trControl = train.control3)
str(newl_train)
stopCluster(cl3)

caret.cv3

preds3 <- predict(caret.cv3, newl_test)
sample_test<-read.csv("test.csv",na.strings=c("","NA"))
submit3 <- data.frame(Loan_ID = sample_test$Loan_ID, Loan_Status = preds3)
write.csv(submit3, file = "sample_submissiona1.csv", row.names = FALSE)

varImp(caret.cv3)
