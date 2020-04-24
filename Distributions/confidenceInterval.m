clear all
clc

%Sample coefficient interval case of Unknown variance (case of known variance)

%Sample problem: Construct a 98% Confidence Interval based on the following
%data: 45, 55, 67, 45, 68, 79, 98, 87, 84, 82 

A = [45, 55, 67, 45, 68, 79, 98, 87, 84, 82];
M = mean(A); %Step1
S = std(A);
df = length(A) - 1; %Step 2

%Confidence level = 98%
a = (1 - 0.95)/2; %Step 3 98%,95%


C1 = tinv([a 1-a],df);%Step4 %critical value
C2 = S/sqrt(length(A));


[M-C1(2)*C2   M+C1(2)*C2]

%Confidence bound on DTC will be wrong because it will be the same over the
%whole cycle but a confidence bound can be constructed on parameters of
%DTC.
