require(MASS);
setwd('/home/rishiraj/cs5011/contest/ml_contest');

data = read.csv('../train_X.csv', header=FALSE);
target = read.csv('../train_Y.csv', header=FALSE);

View(head(data))
dim(data) #14332, 2048
View(data[,1])

complete_data = cbind(data, target)
View(complete_data[,2049])

#split into two: train-test
samp = sort(sample(c(1:14332), round(0.8*14332)))
train_data = complete_data[samp,];
test_data = complete_data[-samp,];

train_data_X = train_data[,-1];
train_data_Y = train_data[,2049];

test_data_X = test_data[,-1];
test_data_Y = test_data[,2049];

write.csv(train_data_X, '../train_data_X.csv', row.names=FALSE);
write.csv(test_data_X, '../test_data_X.csv', row.names=FALSE);
write.csv(train_data_Y, '../train_data_Y.csv', row.names=FALSE);
write.csv(test_data_Y, '../test_data_Y.csv', row.names=FALSE);

# split into 10 for cross validation.
#run following lines 10 times to generate 10 sets
#keep change filenames from 1 to 10 in last two lines

samp = sort(sample(c(1:14332), round(0.2*14332), replace=F))
train_data = complete_data[samp,];
train_data_X = train_data[,-1];
train_data_Y = train_data[,2049];
write.csv(train_data_X, '../train10_X.csv', row.names=FALSE)
write.csv(train_data_Y, '../train10_Y.csv', row.names=FALSE)
