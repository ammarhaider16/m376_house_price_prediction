function [C, RMSE]=training()
data=readtable('train.csv');
X(:,1:5)=data(:,1:5);
X(:,6:12)=data(:,7:13);
Y=data(:,6); % 6th col is median home value
[s,~]=size(X);
X=table2array(X);
Y=table2array(Y);
A=[ones(s,1), X]; % may try different values for intercept 
C=A\Y; % calculate coefficiants
Ypred=A*C; % regression y values
R=Ypred-Y; % R --- residuals
RMSE=sqrt(mean(R.^2)); 
rsquared=1-(sum(R.^2)/sum((Y-mean(Y)).^2));
T1=table(Y,Ypred,'VariableNames',{'y_true','y_pred'});
writetable(T1,'regression_outputs.csv');
T2=table(RMSE,rsquared,'VariableNames',{'RMSE','r_squared'});
writetable(T2,'regression_outputs.csv');
disp(RMSE)
end