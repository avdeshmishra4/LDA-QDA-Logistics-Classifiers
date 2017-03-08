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
											% Hold out D1 for test and train on rest
train_data = [D2;D3;D4;D5;D6;D7;D8;D9;D10];	% concatinate dataset D2 upto D10 vertically to form training data				
[row,col] = size(train_data);				% find size of the training data
X_temp = train_data(:,1:col-1);				% create temporary matrix (X_temp) by removing last column
V_Ones = ones(row,1);						% create a vector with all element 1
X = [V_Ones X_temp];						% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);							% find size of matrix X
Y = train_data(:,col);						% save all the real Y (last column of train_data) into Y vector
Beta = zeros(M,1);							% number of betas = no of features+1, initialize beta vector to zero
W = eye(N,N);								% create an identity matrix (hassian matrix) of size N*N
eta = zeros(N,1);							% create vector eta and initialize it with zeros (stores the probability
for j = 1:20								% loop to calculate beta values iteratively
	for i = 1: N							% for each sample points in training data
		data = X(i,:)';						
		p = exp(data'*Beta)/(1+exp(data'*Beta));	% calculate the probability
		eta(i) = p;							% store the probability into eta vector
		p = p*(1-p);						% calculate p*(1-p) and
		W(i,i) = p;							% store into W matrix (Hassian matrix)
	end

	z = (X*Beta+inv(W)*(Y-eta));			% calculate z => adjusted response
	Beta = inv(X'*W*X)*X'*W*z;				% calculate iteratively weighted beta

end

%******** Testing **********
[row,col] = size(D1);						% calculate size of D1 training set
X_temp = D1(:,1:col-1);						% create a temporary matrix excluding the last column
V_Ones = ones(row,1);						% create a vector with all its element 1
X = [V_Ones X_temp];						% add vector of 1 horizontally with X_temp to create set of features X
[N,M] = size(X);							% calculate size of training set X
Y = D1(:,col);								% store value of real class of D1 into Y vector

incorrect_count = 0;						% variable to store miss-classification
for i=1:N									% for number of samples in test set
	x = X(i,:)';							% take each row and convert it into a column vector 
	y = Y(i,1);								% store the real class into y variable
	pred_y = 0;								% initialize predicted y to zero
	for j = 1:M								% calculate the value of predicted y using beta calculated from above
		pred_y = pred_y+Beta(j,1)*x(j,1);
	end
	if pred_y >= 0							% if pred_y is >= 0 then pred_y = 1 else zero
		pred_y = 1;
	elseif pred_y < 0
		pred_y = 0;
	end
	
	if y ~= pred_y							% if miss-classified increse counter
		incorrect_count = incorrect_count+1;
	end
end
%incorrect count
incorrect_count								% print incorrect_count
%Total test points
N											% print total number of sample points in testset
error_fold1 = (incorrect_count/N)*100		% calculate error rate
%============================= End of Fold#1==============================

%============================= Fold#2==============================
												% Hold out D2 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D2 now
train_data = [D1;D3;D4;D5;D6;D7;D8;D9;D10];
[row,col] = size(train_data);
X_temp = train_data(:,1:col-1);
V_Ones = ones(row,1);							% create a vector with all element 1
X = [V_Ones X_temp];							% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = train_data(:,col);
Beta = zeros(M,1);
W = eye(N,N);
eta = zeros(N,1);
for j = 1:20
	for i = 1: N
		data = X(i,:)';
		p = exp(data'*Beta)/(1+exp(data'*Beta));
		eta(i) = p;
		p = p*(1-p);
		W(i,i) = p;
	end

	z = (X*Beta+inv(W)*(Y-eta));
	Beta = inv(X'*W*X)*X'*W*z;

end

%******** Testing **********
[row,col] = size(D2);
X_temp = D2(:,1:col-1);
V_Ones = ones(row,1);								% create a vector with all element 1
X = [V_Ones X_temp];								% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = D2(:,col);

incorrect_count = 0;
for i=1:N
	x = X(i,:)';									% X is already a row vector here 
	y = Y(i,1);										% mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_y = 0;
	for j = 1:M
		pred_y = pred_y+Beta(j,1)*x(j,1);
	end
	if pred_y >= 0
		pred_y = 1;
	elseif pred_y < 0
		pred_y = 0;
	end
	
	if y ~= pred_y
		incorrect_count = incorrect_count+1;
	end
end
%incorrect count
incorrect_count
%Total test points
N
error_fold2 = (incorrect_count/N)*100
%============================= End of Fold#2==============================

%============================= Fold#3==============================
												% Hold out D3 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D3 now
train_data = [D1;D2;D4;D5;D6;D7;D8;D9;D10];
[row,col] = size(train_data);
X_temp = train_data(:,1:col-1);
V_Ones = ones(row,1);							% create a vector with all element 1
X = [V_Ones X_temp];							% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = train_data(:,col);
Beta = zeros(M,1);
W = eye(N,N);
eta = zeros(N,1);
for j = 1:20
	for i = 1: N
		data = X(i,:)';
		p = exp(data'*Beta)/(1+exp(data'*Beta));
		eta(i) = p;
		p = p*(1-p);
		W(i,i) = p;
	end

	z = (X*Beta+inv(W)*(Y-eta));
	Beta = inv(X'*W*X)*X'*W*z;

end

%******** Testing **********
[row,col] = size(D3);
X_temp = D3(:,1:col-1);
V_Ones = ones(row,1);							% create a vector with all element 1
X = [V_Ones X_temp];							% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = D3(:,col);

incorrect_count = 0;
for i=1:N
	x = X(i,:)';								% X is already a row vector here 
	y = Y(i,1);									% mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_y = 0;
	for j = 1:M
		pred_y = pred_y+Beta(j,1)*x(j,1);
	end
	if pred_y >= 0
		pred_y = 1;
	elseif pred_y < 0
		pred_y = 0;
	end
	
	if y ~= pred_y
		incorrect_count = incorrect_count+1;
	end
end
%incorrect count
incorrect_count
%Total test points
N
error_fold3 = (incorrect_count/N)*100
%============================= End of Fold#3==============================

%============================= Fold#4==============================
												% Hold out D4 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D4 now
train_data = [D1;D2;D3;D5;D6;D7;D8;D9;D10];
[row,col] = size(train_data);
X_temp = train_data(:,1:col-1);
V_Ones = ones(row,1);							% create a vector with all element 1
X = [V_Ones X_temp];							% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = train_data(:,col);
Beta = zeros(M,1);
W = eye(N,N);
eta = zeros(N,1);
for j = 1:20
	for i = 1: N
		data = X(i,:)';
		p = exp(data'*Beta)/(1+exp(data'*Beta));
		eta(i) = p;
		p = p*(1-p);
		W(i,i) = p;
	end

	z = (X*Beta+inv(W)*(Y-eta));
	Beta = inv(X'*W*X)*X'*W*z;

end

%******** Testing **********
[row,col] = size(D4);
X_temp = D4(:,1:col-1);
V_Ones = ones(row,1);							% create a vector with all element 1
X = [V_Ones X_temp];							% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = D4(:,col);

incorrect_count = 0;
for i=1:N
	x = X(i,:)';								% X is already a row vector here 
	y = Y(i,1);									%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_y = 0;
	for j = 1:M
		pred_y = pred_y+Beta(j,1)*x(j,1);
	end
	if pred_y >= 0
		pred_y = 1;
	elseif pred_y < 0
		pred_y = 0;
	end
	
	if y ~= pred_y
		incorrect_count = incorrect_count+1;
	end
end
%incorrect count
incorrect_count
%Total test points
N
error_fold4 = (incorrect_count/N)*100
%============================= End of Fold#4==============================

%============================= Fold#5==============================
												% Hold out D5 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D5 now
train_data = [D1;D2;D3;D4;D6;D7;D8;D9;D10];
[row,col] = size(train_data);
X_temp = train_data(:,1:col-1);
V_Ones = ones(row,1);							% create a vector with all element 1
X = [V_Ones X_temp];							% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = train_data(:,col);
Beta = zeros(M,1);
W = eye(N,N);
eta = zeros(N,1);
for j = 1:20
	for i = 1: N
		data = X(i,:)';
		p = exp(data'*Beta)/(1+exp(data'*Beta));
		eta(i) = p;
		p = p*(1-p);
		W(i,i) = p;
	end

	z = (X*Beta+inv(W)*(Y-eta));
	Beta = inv(X'*W*X)*X'*W*z;

end

%******** Testing **********
[row,col] = size(D5);
X_temp = D5(:,1:col-1);
V_Ones = ones(row,1);								% create a vector with all element 1
X = [V_Ones X_temp];								% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = D5(:,col);

incorrect_count = 0;
for i=1:N
	x = X(i,:)';									% X is already a row vector here 
	y = Y(i,1);										%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_y = 0;
	for j = 1:M
		pred_y = pred_y+Beta(j,1)*x(j,1);
	end
	if pred_y >= 0
		pred_y = 1;
	elseif pred_y < 0
		pred_y = 0;
	end
	
	if y ~= pred_y
		incorrect_count = incorrect_count+1;
	end
end
%incorrect count
incorrect_count
%Total test points
N
error_fold5 = (incorrect_count/N)*100
%============================= End of Fold#5==============================

%============================= Fold#6==============================
												% Hold out D6 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D6 now
train_data = [D1;D2;D3;D4;D5;D7;D8;D9;D10];
[row,col] = size(train_data);
X_temp = train_data(:,1:col-1);
V_Ones = ones(row,1);							% create a vector with all element 1
X = [V_Ones X_temp];							% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = train_data(:,col);
Beta = zeros(M,1);
W = eye(N,N);
eta = zeros(N,1);
for j = 1:20
	for i = 1: N
		data = X(i,:)';
		p = exp(data'*Beta)/(1+exp(data'*Beta));
		eta(i) = p;
		p = p*(1-p);
		W(i,i) = p;
	end

	z = (X*Beta+inv(W)*(Y-eta));
	Beta = inv(X'*W*X)*X'*W*z;

end

%******** Testing **********
[row,col] = size(D6);
X_temp = D6(:,1:col-1);
V_Ones = ones(row,1);							% create a vector with all element 1
X = [V_Ones X_temp];							% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = D6(:,col);

incorrect_count = 0;
for i=1:N
	x = X(i,:)';								% X is already a row vector here 
	y = Y(i,1);									%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_y = 0;
	for j = 1:M
		pred_y = pred_y+Beta(j,1)*x(j,1);
	end
	if pred_y >= 0
		pred_y = 1;
	elseif pred_y < 0
		pred_y = 0;
	end
	
	if y ~= pred_y
		incorrect_count = incorrect_count+1;
	end
end
%incorrect count
incorrect_count
%Total test points
N
error_fold6 = (incorrect_count/N)*100
%============================= End of Fold#6==============================

%============================= Fold#7==============================
												% Hold out D7 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D7 now
train_data = [D1;D2;D3;D4;D5;D6;D8;D9;D10];
[row,col] = size(train_data);
X_temp = train_data(:,1:col-1);
V_Ones = ones(row,1);							% create a vector with all element 1
X = [V_Ones X_temp];							% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = train_data(:,col);
Beta = zeros(M,1);
W = eye(N,N);
eta = zeros(N,1);
for j = 1:20
	for i = 1: N
		data = X(i,:)';
		p = exp(data'*Beta)/(1+exp(data'*Beta));
		eta(i) = p;
		p = p*(1-p);
		W(i,i) = p;
	end

	z = (X*Beta+inv(W)*(Y-eta));
	Beta = inv(X'*W*X)*X'*W*z;

end

%******** Testing **********
[row,col] = size(D7);
X_temp = D7(:,1:col-1);
V_Ones = ones(row,1);							% create a vector with all element 1
X = [V_Ones X_temp];							% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = D7(:,col);

incorrect_count = 0;
for i=1:N
	x = X(i,:)';								% X is already a row vector here 
	y = Y(i,1);									%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_y = 0;
	for j = 1:M
		pred_y = pred_y+Beta(j,1)*x(j,1);
	end
	if pred_y >= 0
		pred_y = 1;
	elseif pred_y < 0
		pred_y = 0;
	end
	
	if y ~= pred_y
		incorrect_count = incorrect_count+1;
	end
end
%incorrect count
incorrect_count
%Total test points
N
error_fold7 = (incorrect_count/N)*100
%============================= End of Fold#7==============================

%============================= Fold#8==============================
												% Hold out D8 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D8 now
train_data = [D1;D2;D3;D4;D5;D6;D7;D9;D10];
[row,col] = size(train_data);
X_temp = train_data(:,1:col-1);
V_Ones = ones(row,1);							% create a vector with all element 1
X = [V_Ones X_temp];							% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = train_data(:,col);
Beta = zeros(M,1);
W = eye(N,N);
eta = zeros(N,1);
for j = 1:20
	for i = 1: N
		data = X(i,:)';
		p = exp(data'*Beta)/(1+exp(data'*Beta));
		eta(i) = p;
		p = p*(1-p);
		W(i,i) = p;
	end

	z = (X*Beta+inv(W)*(Y-eta));
	Beta = inv(X'*W*X)*X'*W*z;

end

%******** Testing **********
[row,col] = size(D8);
X_temp = D8(:,1:col-1);
V_Ones = ones(row,1);								% create a vector with all element 1
X = [V_Ones X_temp];								% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = D8(:,col);

incorrect_count = 0;
for i=1:N
	x = X(i,:)';									% X is already a row vector here 
	y = Y(i,1);										%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_y = 0;
	for j = 1:M
		pred_y = pred_y+Beta(j,1)*x(j,1);
	end
	if pred_y >= 0
		pred_y = 1;
	elseif pred_y < 0
		pred_y = 0;
	end
	
	if y ~= pred_y
		incorrect_count = incorrect_count+1;
	end
end
%incorrect count
incorrect_count
%Total test points
N
error_fold8 = (incorrect_count/N)*100
%============================= End of Fold#8==============================

%============================= Fold#9==============================
												% Hold out D9 and train on rest
												% All of the procedure is same as above besides, the test
												% dataset is D9 now
train_data = [D1;D2;D3;D4;D5;D6;D7;D8;D10];
[row,col] = size(train_data);
X_temp = train_data(:,1:col-1);
V_Ones = ones(row,1);							% create a vector with all element 1
X = [V_Ones X_temp];							% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = train_data(:,col);
Beta = zeros(M,1);
W = eye(N,N);
eta = zeros(N,1);
for j = 1:20
	for i = 1: N
		data = X(i,:)';
		p = exp(data'*Beta)/(1+exp(data'*Beta));
		eta(i) = p;
		p = p*(1-p);
		W(i,i) = p;
	end

	z = (X*Beta+inv(W)*(Y-eta));
	Beta = inv(X'*W*X)*X'*W*z;

end

%******** Testing **********
[row,col] = size(D9);
X_temp = D9(:,1:col-1);
V_Ones = ones(row,1);							% create a vector with all element 1
X = [V_Ones X_temp];							% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = D9(:,col);

incorrect_count = 0;
for i=1:N
	x = X(i,:)';								% X is already a row vector here 
	y = Y(i,1);									%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_y = 0;
	for j = 1:M
		pred_y = pred_y+Beta(j,1)*x(j,1);
	end
	if pred_y >= 0
		pred_y = 1;
	elseif pred_y < 0
		pred_y = 0;
	end
	
	if y ~= pred_y
		incorrect_count = incorrect_count+1;
	end
end
%incorrect count
incorrect_count
%Total test points
N
error_fold9 = (incorrect_count/N)*100
%============================= End of Fold#9==============================

%============================= Fold#10==============================
											% Hold out D10 and train on rest
											% All of the procedure is same as above besides, the test
											% dataset is D10 now
train_data = [D1;D2;D3;D4;D5;D6;D7;D8;D9];
[row,col] = size(train_data);
X_temp = train_data(:,1:col-1);
V_Ones = ones(row,1);						% create a vector with all element 1
X = [V_Ones X_temp];						% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = train_data(:,col);
Beta = zeros(M,1);
W = eye(N,N);
eta = zeros(N,1);
for j = 1:20
	for i = 1: N
		data = X(i,:)';
		p = exp(data'*Beta)/(1+exp(data'*Beta));
		eta(i) = p;
		p = p*(1-p);
		W(i,i) = p;
	end

	z = (X*Beta+inv(W)*(Y-eta));
	Beta = inv(X'*W*X)*X'*W*z;

end

%******** Testing **********
[row,col] = size(D10);
X_temp = D10(:,1:col-1);
V_Ones = ones(row,1);							% create a vector with all element 1
X = [V_Ones X_temp];							% this adds extra column with all element "1" at the begining of the matrix X_temp to create matrix "X"
[N,M] = size(X);
Y = D10(:,col);

incorrect_count = 0;
for i=1:N
	x = X(i,:)';								% X is already a row vector here 
	y = Y(i,1);									%  mean_class0 and mean_class1 is also a row vector, so do inverse of what is in formula
	pred_y = 0;
	for j = 1:M
		pred_y = pred_y+Beta(j,1)*x(j,1);
	end
	if pred_y >= 0
		pred_y = 1;
	elseif pred_y < 0
		pred_y = 0;
	end
	
	if y ~= pred_y
		incorrect_count = incorrect_count+1;
	end
end
%incorrect count
incorrect_count
%Total test points
N
error_fold10 = (incorrect_count/N)*100
%============================= End of Fold#10==============================
Logistic_avg_error_rate_10Folds = (error_fold1+error_fold2+error_fold3+error_fold4+error_fold5+error_fold6+error_fold7+error_fold8+error_fold9+error_fold10)/10
% Logistic_avg_error_rate_10Folds = 27.2479%