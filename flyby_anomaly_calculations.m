data = csvread('data.csv');

%observed anomalous velocity change
y = data(:, 7);

%raw data
x = data(:, 1:6);
x1_ngtve = x.^-1;
x2_ngtve = x.^-2;
x3_ngtve = x.^-3;

%anomalous velocity change predicted by Acedo at al.
f = (16.2247*10^(-6)).*data(:,4).*(((6371)./(6371+data(:,3))).^(21.8739)).*(abs(cosd(2*data(:,1))).^(1.1537)).*(sind(2*data(:,2)))*10^(6);

% mean square error of original equation
disp('original equation MSE')
disp(immse(f,lab))

% regularized linear regression coefficients all go to 0
[B, stats] = lasso(x,y, 'CV', 9, 'Alpha', 1);
coefs1 = B(:,stats.IndexMinMSE)';
disp('regularized linear regression MSE')
disp(stats.MSE(stats.IndexMinMSE))
disp('coefficients')
disp(coefs1)

% regularized linear regression of inverse of all predictors sets
[B, stats] = lasso(x1_ngtve,y, 'CV', 9, 'Alpha', 1);
coefs2 = B(:,stats.IndexMinMSE)';
disp('regularized linear regression MSE (predictor inverses)')
disp(stats.MSE(stats.IndexMinMSE))
disp('coefficients')
disp(coefs2)

% regularized linear regression of cubic inverse of all predictors; sets
% azimuthal velocity and altitude to non-zero value; HAS MINIMUM MSE
[B, stats] = lasso(x2_ngtve,y, 'CV', 9, 'Alpha', 1);
coefs3 = B(:,stats.IndexMinMSE)';
disp('regularized linear regression MSE (predictor square inverses)')
disp(stats.MSE(stats.IndexMinMSE))
disp('coefficients')
disp(coefs3)

% regularized linear regression of cubic inverse of all predictors
[B, stats] = lasso(x3_ngtve,y, 'CV', 9, 'Alpha', 1);
coefs4 = B(:,stats.IndexMinMSE)';
disp('regularized linear regression MSE (predictor cubic inverses)')
disp(stats.MSE(stats.IndexMinMSE))
disp('coefficients')
disp(coefs4)

%equation 7) | fluid density as a function of height (density equation emperically derived from MSIS-E-90 Atmosphere Model)
p = data(:,3).^7.172;
vp = x2_ngtve(:,5).*p;
x_2 = [vp, f];
coefs5 = regress(y, x_2);
disp('equation 7 MSE')
disp(immse(x_2*coefs5,y))

%equation 10) | fluid density as a function of inclination angle 
p2 = (cos(data(:,1)).^-1);
vp2 = x2_ngtve(:,5).*p2;
x_3 = [vp2, f];
coefs6 = regress(y, x_3);
disp('equation 10 coefficients')
disp(coefs6)
disp('equation 10 MSE')
disp(immse(x_3*coefs6,y))
disp('equation 10 predicted anomalous velocity change')
disp(x_3*coefs6)
