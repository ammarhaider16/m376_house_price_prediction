function [C,RMSE]=testing(cvector)
data=readtable('test.csv');
X(:,1:5)=data(:,1:5);
X(:,6:12)=data(:,7:13);
Y=data(:,6); % 6th col is median home value
[s,~]=size(X); 
X=table2array(X);
Y=table2array(Y);
A=[ones(s,1), X]; % intercepts =1, could mess around with
C=cvector; % use c vals calculated by trained model
Ypred=A*C; % regression y values
R=Ypred-Y; % residuals
RMSE=sqrt(mean(R.^2)); 
rsquared=1-(sum(R.^2)/sum((Y-mean(Y)).^2));
T1=table(Y,Ypred,'VariableNames',{'y_true','y_pred'});
writetable(T1,'regression_test_outputs.csv');
T2=table(RMSE,rsquared,'VariableNames',{'RMSE','r_squared'});
writetable(T2,'regression_test_outputs.csv');
end 