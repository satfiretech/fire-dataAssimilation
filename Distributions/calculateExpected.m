%Order statistics
%Calculate the kth order statistic expectd value [Expected kth order statistic]. The density is found in
%notes (orderStatSum.pdf) by William Wu
%fk with n (number of samples), k (kth order)

% clear all
% clc
% 
% 
% Q = quadl(@integrand,0,100)


clear all
clc



n = 24; %Number of samples
k = 1;  %kth order

syms x

%Gumbel

%Fe = 1 - exp(x) + exp(2*x)/2 - exp(3*x)/6 + exp(4*x)/factorial(4) - exp(5*x)/factorial(5) + exp(6*x)/factorial(6) - exp(7*x)/factorial(7);%exp(-exp(x));

% f = exp(x)*exp(-exp(x));
% F1 = 1 - exp(-exp(x));
% F2 = exp(-exp(x));

Fe = exp(-exp(x)); %Negative skewed Gumbel

f = exp(x)*Fe;
F1 = 1 - Fe;
F2 = Fe;

% Fe = exp(-exp(-x)); %POsitive skewed Gumbel
% 
% f = exp(-x)*Fe;
% F1 = 1 - Fe;
% F2 = Fe;




fk = factorial(n)/(factorial(k-1)*factorial(n-k)) * f * (F1)^(k-1) * (F2)^(n-k); %orderStatSum.pdf by William Wu
%fk = factorial(n)/(factorial(k-1)*factorial(n-k)) * f * (F1)^(k-1) * (exp(-(n-k)*exp(x)));


EG = int(x*fk,-Inf,Inf)

%Normal


f = 1/sqrt(2*pi) * exp(-x^2/2);
F1 = 1/2*(1 + erf(x/sqrt(2)));
F2 = 1/2*(1 - erf(x/sqrt(2)));

fk = factorial(n)/(factorial(k-1)*factorial(n-k)) * f * (F1)^(k-1) * (F2)^(n-k);


EN = int(x*fk,-Inf,Inf)


%Logistic


f = exp(-x)/(1 + exp(-x))^2;
F1 = 1/(1 + exp(-x));
F2 = exp(-x)/(1 + exp(-x));

fk = factorial(n)/(factorial(k-1)*factorial(n-k)) * f * (F1)^(k-1) * (F2)^(n-k);


EL = int(x*fk,-Inf,Inf)