%1st, 2nd and 3rd order Gaussian statistics
close all
clc


x = -5:0.00001:5;

y = 1/sqrt(2*pi)*exp(-x.^2/2);

% z1 = 1 + erf(x/sqrt(2));
% z2 = 1 - erf(x/sqrt(2));
% 
% figure
% plot(x,y)
% 
% figure
% plot(x,y.*z1)
% 
% figure
% plot(x,y.*z2)
% 
% 
% H = [1 -0.506;1 0.506];
% V = [1/sqrt(2) 0;0 1/sqrt(2)];
% inv(H'*inv(V)*H)*H'*inv(V)


F1 = 1/2*(1 + erf(x/sqrt(2)));
F2 = 1/2*(1 - erf(x/sqrt(2)));


z1 = 3*F2.^2;
z2 = 6*F1.*F2;
z3 = 3*F2.^2;


figure
plot(x,y)

figure
plot(x,y.*z1)


figure
plot(x,y.*z2)


figure
plot(x,y.*z3)



