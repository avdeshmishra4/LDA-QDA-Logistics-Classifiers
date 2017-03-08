load inputDataset.txt							%load the input
[m,n] = size(inputDataset);						%find number of rows and column in input
D1 = [];										%create an empty matrix
D2 = [];
D3 = [];
D4 = [];
D5 = [];
D6 = [];
D7 = [];
D8 = [];
D9 = [];
D10 = [];
for i = 1: m									%split input data into 10 subset, using the modular
	value = mod(i,10);							%operation each sample point is assigned to D1 to D10 subset  
	switch value
    case 0
        x = inputDataset(i,:);
		D1 = [D1;x];
    case 1
        x = inputDataset(i,:);
		D2 = [D2;x];
    case 2
        x = inputDataset(i,:);
		D3 = [D3;x];
	case 3
        x = inputDataset(i,:);
		D4 = [D4;x];
    case 4
        x = inputDataset(i,:);
		D5 = [D5;x];
    case 5
        x = inputDataset(i,:);
		D6 = [D6;x];
	case 6
        x = inputDataset(i,:);
		D7 = [D7;x];
    case 7
        x = inputDataset(i,:);
		D8 = [D8;x];
    case 8
        x = inputDataset(i,:);
		D9 = [D9;x];
	otherwise
        x = inputDataset(i,:);
		D10 = [D10;x];
	end
end
%============================= Fold#1==============================
												% Hold out D1 and train on rest
train_data = [D2;D3;D4;D5;D6;D7;D8;D9;D10];		% concatinate dataset D2 upto D10 vertically to form training data				
[row,col] = size(train_data);					% find size of the training data
class1 = [];									% create empty matrix
class0 = [];
for i = 1:row									% store train dataset into new matrix class1 and class0
	if train_data(i,col) == 1					% based on the last column (class type) of training data
		data = train_data(i,1:9);
		class1 = [class1;data];
	elseif train_data(i,col) == 0
		data = train_data(i,1:9);
		class0 = [class0;data];
	end
end
[A,B] = size(class1);							% find size of class1
prob_class1 = A/row;							% find probability of class1
[C,D] = size(class0);							% find size of class0
prob_class0 = C/row;							% find probability of class0

mean_class1 = mean(class1)						% calculates mean for each feature for class = 1
mean_class0 = mean(class0)						% calculates mean for each feature for class = 0

sigma_class1 = cov(class1)						% calculates covariance of class = 1
sigma_class0 = cov(class0)						% calculates covariance of class = 0



[row,col] = size(D1);							% calculate size of test set D1
incorrect_count = 0;							% counter to store number of miss-classification while testing
for i=1:row										% test on number of sample points of test data (D1)
	X = D1(i,1:9);								% X is already a row vector here 
	Y = D1(i,col);								%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_Y = -1;
	QDelta0 = -0.5*log(det(sigma_class0))-0.5*(X'-mean_class0')'*inv(sigma_class0)*(X'-mean_class0')+log(prob_class0);
	QDelta1 = -0.5*log(det(sigma_class1))-0.5*(X'-mean_class1')'*inv(sigma_class1)*(X'-mean_class1')+log(prob_class1);
	if QDelta0 > QDelta1						% if QDelta0 is greater classify as class 0 else class 1
		pred_Y = 0;
	elseif QDelta0 < QDelta1
		pred_Y = 1;
	elseif QDelta0 == QDelta1
		pred_Y = 0;
	end
	
	if Y ~= pred_Y								% check value of pred_Y with real Y, if not equal, increase miss-classified count
		incorrect_count = incorrect_count+1;
	end
end

error_fold1 = (incorrect_count/row)*100			% error rate = (number_of_miss-classified/no of data points in test set) * 100%

%============================= End of Fold#1==============================

%============================= Fold#2==============================
												% Hold out D2 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D2 now
train_data = [D1;D3;D4;D5;D6;D7;D8;D9;D10];
[row,col] = size(train_data);
class1 = [];
class0 = [];
for i = 1:row
	if train_data(i,col) == 1
		data = train_data(i,1:9);
		class1 = [class1;data];
	elseif train_data(i,col) == 0
		data = train_data(i,1:9);
		class0 = [class0;data];
	end
end
[A,B] = size(class1);	
prob_class1 = A/row;
[C,D] = size(class0);	
prob_class0 = C/row;

mean_class1 = mean(class1)		% calculates mean for each feature for class = 1
mean_class0 = mean(class0)		% calculates mean for each feature for class = 0

sigma_class1 = cov(class1)		% calculates covariance of class = 1
sigma_class0 = cov(class0)		% calculates covariance of class = 0



[row,col] = size(D2);
incorrect_count = 0;
for i=1:row
	X = D2(i,1:9);	% X is already a row vector here 
	Y = D2(i,col);	%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_Y = -1;
	QDelta0 = -0.5*log(det(sigma_class0))-0.5*(X'-mean_class0')'*inv(sigma_class0)*(X'-mean_class0')+log(prob_class0);
	QDelta1 = -0.5*log(det(sigma_class1))-0.5*(X'-mean_class1')'*inv(sigma_class1)*(X'-mean_class1')+log(prob_class1);
	if QDelta0 > QDelta1
		pred_Y = 0;
	elseif QDelta0 < QDelta1
		pred_Y = 1;
	elseif QDelta0 == QDelta1
		pred_Y = 0;
	end
	
	if Y ~= pred_Y
		incorrect_count = incorrect_count+1;
	end
end

error_fold2 = (incorrect_count/row)*100

%============================= End of Fold#2==============================

%============================= Fold#3==============================
												% Hold out D3 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D3 now
train_data = [D1;D2;D4;D5;D6;D7;D8;D9;D10];
[row,col] = size(train_data);
class1 = [];
class0 = [];
for i = 1:row
	if train_data(i,col) == 1
		data = train_data(i,1:9);
		class1 = [class1;data];
	elseif train_data(i,col) == 0
		data = train_data(i,1:9);
		class0 = [class0;data];
	end
end
[A,B] = size(class1);	
prob_class1 = A/row;
[C,D] = size(class0);	
prob_class0 = C/row;

mean_class1 = mean(class1)		% calculates mean for each feature for class = 1
mean_class0 = mean(class0)		% calculates mean for each feature for class = 0

sigma_class1 = cov(class1)		% calculates covariance of class = 1
sigma_class0 = cov(class0)		% calculates covariance of class = 0



[row,col] = size(D3);
incorrect_count = 0;
for i=1:row
	X = D3(i,1:9);	% X is already a row vector here 
	Y = D3(i,col);	%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_Y = -1;
	QDelta0 = -0.5*log(det(sigma_class0))-0.5*(X'-mean_class0')'*inv(sigma_class0)*(X'-mean_class0')+log(prob_class0);
	QDelta1 = -0.5*log(det(sigma_class1))-0.5*(X'-mean_class1')'*inv(sigma_class1)*(X'-mean_class1')+log(prob_class1);
	if QDelta0 > QDelta1
		pred_Y = 0;
	elseif QDelta0 < QDelta1
		pred_Y = 1;
	elseif QDelta0 == QDelta1
		pred_Y = 0;
	end
	
	if Y ~= pred_Y
		incorrect_count = incorrect_count+1;
	end
end

error_fold3 = (incorrect_count/row)*100

%============================= End of Fold#3==============================

%============================= Fold#4==============================
												% Hold out D4 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D4 now
train_data = [D1;D2;D3;D5;D6;D7;D8;D9;D10];
[row,col] = size(train_data);
class1 = [];
class0 = [];
for i = 1:row
	if train_data(i,col) == 1
		data = train_data(i,1:9);
		class1 = [class1;data];
	elseif train_data(i,col) == 0
		data = train_data(i,1:9);
		class0 = [class0;data];
	end
end
[A,B] = size(class1);	
prob_class1 = A/row;
[C,D] = size(class0);	
prob_class0 = C/row;

mean_class1 = mean(class1)		% calculates mean for each feature for class = 1
mean_class0 = mean(class0)		% calculates mean for each feature for class = 0

sigma_class1 = cov(class1)		% calculates covariance of class = 1
sigma_class0 = cov(class0)		% calculates covariance of class = 0



[row,col] = size(D4);
incorrect_count = 0;
for i=1:row
	X = D4(i,1:9);	% X is already a row vector here 
	Y = D4(i,col);	%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_Y = -1;
	QDelta0 = -0.5*log(det(sigma_class0))-0.5*(X'-mean_class0')'*inv(sigma_class0)*(X'-mean_class0')+log(prob_class0);
	QDelta1 = -0.5*log(det(sigma_class1))-0.5*(X'-mean_class1')'*inv(sigma_class1)*(X'-mean_class1')+log(prob_class1);
	if QDelta0 > QDelta1
		pred_Y = 0;
	elseif QDelta0 < QDelta1
		pred_Y = 1;
	elseif QDelta0 == QDelta1
		pred_Y = 0;
	end
	
	if Y ~= pred_Y
		incorrect_count = incorrect_count+1;
	end
end

error_fold4 = (incorrect_count/row)*100

%============================= End of Fold#4==============================

%============================= Fold#5==============================
												% Hold out D5 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D5 now
train_data = [D1;D2;D3;D4;D6;D7;D8;D9;D10];
[row,col] = size(train_data);
class1 = [];
class0 = [];
for i = 1:row
	if train_data(i,col) == 1
		data = train_data(i,1:9);
		class1 = [class1;data];
	elseif train_data(i,col) == 0
		data = train_data(i,1:9);
		class0 = [class0;data];
	end
end
[A,B] = size(class1);	
prob_class1 = A/row;
[C,D] = size(class0);	
prob_class0 = C/row;

mean_class1 = mean(class1)		% calculates mean for each feature for class = 1
mean_class0 = mean(class0)		% calculates mean for each feature for class = 0

sigma_class1 = cov(class1)		% calculates covariance of class = 1
sigma_class0 = cov(class0)		% calculates covariance of class = 0



[row,col] = size(D5);
incorrect_count = 0;
for i=1:row
	X = D5(i,1:9);	% X is already a row vector here 
	Y = D5(i,col);	%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_Y = -1;
	QDelta0 = -0.5*log(det(sigma_class0))-0.5*(X'-mean_class0')'*inv(sigma_class0)*(X'-mean_class0')+log(prob_class0);
	QDelta1 = -0.5*log(det(sigma_class1))-0.5*(X'-mean_class1')'*inv(sigma_class1)*(X'-mean_class1')+log(prob_class1);
	if QDelta0 > QDelta1
		pred_Y = 0;
	elseif QDelta0 < QDelta1
		pred_Y = 1;
	elseif QDelta0 == QDelta1
		pred_Y = 0;
	end
	
	if Y ~= pred_Y
		incorrect_count = incorrect_count+1;
	end
end

error_fold5 = (incorrect_count/row)*100

%============================= End of Fold#5==============================

%============================= Fold#6==============================
												% Hold out D6 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D6 now
train_data = [D1;D2;D3;D4;D5;D7;D8;D9;D10];
[row,col] = size(train_data);
class1 = [];
class0 = [];
for i = 1:row
	if train_data(i,col) == 1
		data = train_data(i,1:9);
		class1 = [class1;data];
	elseif train_data(i,col) == 0
		data = train_data(i,1:9);
		class0 = [class0;data];
	end
end
[A,B] = size(class1);	
prob_class1 = A/row;
[C,D] = size(class0);	
prob_class0 = C/row;

mean_class1 = mean(class1)		% calculates mean for each feature for class = 1
mean_class0 = mean(class0)		% calculates mean for each feature for class = 0

sigma_class1 = cov(class1)		% calculates covariance of class = 1
sigma_class0 = cov(class0)		% calculates covariance of class = 0



[row,col] = size(D6);
incorrect_count = 0;
for i=1:row
	X = D6(i,1:9);	% X is already a row vector here 
	Y = D6(i,col);	%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_Y = -1;
	QDelta0 = -0.5*log(det(sigma_class0))-0.5*(X'-mean_class0')'*inv(sigma_class0)*(X'-mean_class0')+log(prob_class0);
	QDelta1 = -0.5*log(det(sigma_class1))-0.5*(X'-mean_class1')'*inv(sigma_class1)*(X'-mean_class1')+log(prob_class1);
	if QDelta0 > QDelta1
		pred_Y = 0;
	elseif QDelta0 < QDelta1
		pred_Y = 1;
	elseif QDelta0 == QDelta1
		pred_Y = 0;
	end
	
	if Y ~= pred_Y
		incorrect_count = incorrect_count+1;
	end
end

error_fold6 = (incorrect_count/row)*100

%============================= End of Fold#6==============================

%============================= Fold#7==============================
												% Hold out D7 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D7 now
train_data = [D1;D2;D3;D4;D5;D6;D8;D9;D10];
[row,col] = size(train_data);
class1 = [];
class0 = [];
for i = 1:row
	if train_data(i,col) == 1
		data = train_data(i,1:9);
		class1 = [class1;data];
	elseif train_data(i,col) == 0
		data = train_data(i,1:9);
		class0 = [class0;data];
	end
end
[A,B] = size(class1);	
prob_class1 = A/row;
[C,D] = size(class0);	
prob_class0 = C/row;

mean_class1 = mean(class1)		% calculates mean for each feature for class = 1
mean_class0 = mean(class0)		% calculates mean for each feature for class = 0

sigma_class1 = cov(class1)		% calculates covariance of class = 1
sigma_class0 = cov(class0)		% calculates covariance of class = 0



[row,col] = size(D7);
incorrect_count = 0;
for i=1:row
	X = D7(i,1:9);	% X is already a row vector here 
	Y = D7(i,col);	%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_Y = -1;
	QDelta0 = -0.5*log(det(sigma_class0))-0.5*(X'-mean_class0')'*inv(sigma_class0)*(X'-mean_class0')+log(prob_class0);
	QDelta1 = -0.5*log(det(sigma_class1))-0.5*(X'-mean_class1')'*inv(sigma_class1)*(X'-mean_class1')+log(prob_class1);
	if QDelta0 > QDelta1
		pred_Y = 0;
	elseif QDelta0 < QDelta1
		pred_Y = 1;
	elseif QDelta0 == QDelta1
		pred_Y = 0;
	end
	
	if Y ~= pred_Y
		incorrect_count = incorrect_count+1;
	end
end

error_fold7 = (incorrect_count/row)*100

%============================= End of Fold#7==============================

%============================= Fold#8==============================
												% Hold out D8 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D8 now
train_data = [D1;D2;D3;D4;D5;D6;D7;D9;D10];
[row,col] = size(train_data);
class1 = [];
class0 = [];
for i = 1:row
	if train_data(i,col) == 1
		data = train_data(i,1:9);
		class1 = [class1;data];
	elseif train_data(i,col) == 0
		data = train_data(i,1:9);
		class0 = [class0;data];
	end
end
[A,B] = size(class1);	
prob_class1 = A/row;
[C,D] = size(class0);	
prob_class0 = C/row;

mean_class1 = mean(class1)		% calculates mean for each feature for class = 1
mean_class0 = mean(class0)		% calculates mean for each feature for class = 0

sigma_class1 = cov(class1)		% calculates covariance of class = 1
sigma_class0 = cov(class0)		% calculates covariance of class = 0



[row,col] = size(D8);
incorrect_count = 0;
for i=1:row
	X = D8(i,1:9);	% X is already a row vector here 
	Y = D8(i,col);	%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_Y = -1;
	QDelta0 = -0.5*log(det(sigma_class0))-0.5*(X'-mean_class0')'*inv(sigma_class0)*(X'-mean_class0')+log(prob_class0);
	QDelta1 = -0.5*log(det(sigma_class1))-0.5*(X'-mean_class1')'*inv(sigma_class1)*(X'-mean_class1')+log(prob_class1);
	if QDelta0 > QDelta1
		pred_Y = 0;
	elseif QDelta0 < QDelta1
		pred_Y = 1;
	elseif QDelta0 == QDelta1
		pred_Y = 0;
	end
	
	if Y ~= pred_Y
		incorrect_count = incorrect_count+1;
	end
end

error_fold8 = (incorrect_count/row)*100

%============================= End of Fold#8==============================

%============================= Fold#9==============================
												% Hold out D9 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D9 now
train_data = [D1;D2;D3;D4;D5;D6;D7;D8;D10];
[row,col] = size(train_data);
class1 = [];
class0 = [];
for i = 1:row
	if train_data(i,col) == 1
		data = train_data(i,1:9);
		class1 = [class1;data];
	elseif train_data(i,col) == 0
		data = train_data(i,1:9);
		class0 = [class0;data];
	end
end
[A,B] = size(class1);	
prob_class1 = A/row;
[C,D] = size(class0);	
prob_class0 = C/row;

mean_class1 = mean(class1)		% calculates mean for each feature for class = 1
mean_class0 = mean(class0)		% calculates mean for each feature for class = 0

sigma_class1 = cov(class1)		% calculates covariance of class = 1
sigma_class0 = cov(class0)		% calculates covariance of class = 0



[row,col] = size(D9);
incorrect_count = 0;
for i=1:row
	X = D9(i,1:9);	% X is already a row vector here 
	Y = D9(i,col);	%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_Y = -1;
	QDelta0 = -0.5*log(det(sigma_class0))-0.5*(X'-mean_class0')'*inv(sigma_class0)*(X'-mean_class0')+log(prob_class0);
	QDelta1 = -0.5*log(det(sigma_class1))-0.5*(X'-mean_class1')'*inv(sigma_class1)*(X'-mean_class1')+log(prob_class1);
	if QDelta0 > QDelta1
		pred_Y = 0;
	elseif QDelta0 < QDelta1
		pred_Y = 1;
	elseif QDelta0 == QDelta1
		pred_Y = 0;
	end
	
	if Y ~= pred_Y
		incorrect_count = incorrect_count+1;
	end
end

error_fold9 = (incorrect_count/row)*100

%============================= End of Fold#8==============================

%============================= Fold#10==============================
											% Hold out D10 and train on rest
											% All of the procedure is same as above besides, the test
											% dataset is D10 now
train_data = [D1;D2;D3;D4;D5;D6;D7;D8;D9];
[row,col] = size(train_data);
class1 = [];
class0 = [];
for i = 1:row
	if train_data(i,col) == 1
		data = train_data(i,1:9);
		class1 = [class1;data];
	elseif train_data(i,col) == 0
		data = train_data(i,1:9);
		class0 = [class0;data];
	end
end
[A,B] = size(class1);	
prob_class1 = A/row;
[C,D] = size(class0);	
prob_class0 = C/row;

mean_class1 = mean(class1)		% calculates mean for each feature for class = 1
mean_class0 = mean(class0)		% calculates mean for each feature for class = 0

sigma_class1 = cov(class1)		% calculates covariance of class = 1
sigma_class0 = cov(class0)		% calculates covariance of class = 0



[row,col] = size(D10);
incorrect_count = 0;
for i=1:row
	X = D10(i,1:9);	% X is already a row vector here 
	Y = D10(i,col);	%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_Y = -1;
	QDelta0 = -0.5*log(det(sigma_class0))-0.5*(X'-mean_class0')'*inv(sigma_class0)*(X'-mean_class0')+log(prob_class0);
	QDelta1 = -0.5*log(det(sigma_class1))-0.5*(X'-mean_class1')'*inv(sigma_class1)*(X'-mean_class1')+log(prob_class1);
	if QDelta0 > QDelta1
		pred_Y = 0;
	elseif QDelta0 < QDelta1
		pred_Y = 1;
	elseif QDelta0 == QDelta1
		pred_Y = 0;
	end
	
	if Y ~= pred_Y
		incorrect_count = incorrect_count+1;
	end
end

error_fold10 = (incorrect_count/row)*100

%============================= End of Fold#10==============================

QDA_avg_error_rate_10Folds = (error_fold1+error_fold2+error_fold3+error_fold4+error_fold5+error_fold6+error_fold7+error_fold8+error_fold9+error_fold10)/10
% avg_error_10Fold =  29.6300%