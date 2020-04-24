function [c1 To Ta w1 w2 tm ts k ITERATIONS_A FCTCOUNT_A]= diurnal_g(b,daylength,phi,tm,ts,TIME,SAMPLE_STD,ErrFct)
        

%tm = 12.00;
%ts = 15.00;
%startt = 6.00;
%n= The day of the year
%b= The training cycle
%phi= Latitude
%tm= time at the maximum temperature
%ts= time at start of exponential decreasing
%startt= the start of the cycle
%diurnal_g.m
%F.-M. Göttsche, F.S. Olesen,  “Modeling of diurnal cycles of brightness temperature extracted from 
%METEOSAT data,”  Remote Sensing of Environment, vol. 76, 2001, pp. 337 – 348. 


%load pix_42055.c6_h1.txt
%a = pix_42055(:,2); %Brightness temperature data
%c = pix_42055(:,3); %Cloud mask

%b = a(97:192);  %Brightness temperature for the daily cycle of interest
%mc = c(97:192); %Cloud mask for the daily cycle of interest


%Initialisation of parameters
To = b(1);%min(b);             %Initialisation of To
Ta = max(b) - b(1); %max(b)-min(b);      %Initialisation of Ta
%dT = b(1) - b(end); %0;                  %Initialisation of dT


%The value of w
w = daylength;
w1 = w;
w2 = w;

%Calculation of the exponential damping factor(initialisation)
% k = w/pi*(atan(pi/w*(ts-tm))-dT/Ta*asin(pi/w*(ts-tm)));


%Nelder-Mead
options = optimset('Display','off','TolX',[],'TolFun',[],'MaxIter',[],'MaxFunEvals',[],'FunValCheck','on');%,'MaxFunEvals',200);  %,'TolX',1e-20); MaxIter and/or MaxFunEvals
%Levenberg-Marquardt, when off, it is Gauss-Newton
%options = optimset('Display','off','TolX',[],'TolFun',[],'MaxIter',1000,'MaxFunEvals',[],'FunValCheck','on','LargeScale','off','LevenbergMarquardt','on');%,'MaxFunEvals',200);  %,'TolX',1e-20); MaxIter and/or MaxFunEvals

% pp = 0;
% save zeropp pp
% 
% sg = 20;
% save savesigma sg

%Day time considered
t = TIME;

p0 = [To ts tm Ta w1 w2];                %Initial values of cosine model parameters

% figure
% hold on

%Nelder-Mead
switch ErrFct %Error Function
    case 'L', %Least squares
        [p,FVAL,EXITFLAG,OUTPUT] = fminsearch(@(p) nerror_v(b.',TIME,p),p0,options);   %Nelder & Mead's simplex method for error minimisation
        %[p,FVAL,EXITFLAG,OUTPUT] = fminsearch(@(p) robustError_g(b.',w,TIME,p),p0,options);   %Nelder & Mead's simplex method for error minimisation
    case 'C',
        [p,FVAL,EXITFLAG,OUTPUT] = fminsearch(@(p) nerror_Chi_v(b.',TIME,SAMPLE_STD,p),p0,options);   %Nelder & Mead's simplex method for error minimisation
    case 'R', %Robust
        global FLAG
        global ROB_SIGMA
        FLAG = 0;
        ROB_SIGMA = 1;
        [p,FVAL,EXITFLAG,OUTPUT] = fminsearch(@(p) robustError_v(b.',TIME,p),p0,options);   %Nelder & Mead's simplex method for error minimisation
    otherwise,
        disp('Error Function: Specify the right error function');
        error('Specify the right error function');
end


ITERATIONS_A = OUTPUT.iterations;
FCTCOUNT_A = OUTPUT.funcCount;

%Levenberg-Marquardt
%[p,RESNORM,RESIDUAL,EXITFLAG,OUTPUT,LAMBDA] = lsqnonlin(@(p) nerror_lsqnl_g(p,b.',w,TIME),p0,[],[],options);   %Nelder & Mead's simplex method for error minimisation
%[p,RESNORM,RESIDUAL,EXITFLAG,OUTPUT,LAMBDA] = lsqnonlin(@(p) nerror_lsqnlChi_g(p,b.',w,TIME,SAMPLE_STD),p0,[],[],options);   %Nelder & Mead's simplex method for error minimisation
%LAMBDA

%LMFnlsq
%[p ssq cnt nfj] = LMFnlsq(@(p) nerror_LMF_g(p,b.',w,TIME),p0);%,'Display','off','XTol',[],'FunTol',[],'MaxIter',1000,'Lambda',[]);%,'MaxFunEvals',200);
%[p ssq cnt nfj] = LMFnlsq2(@(p) nerror_LMF_g(p,b.',w,TIME),p0);%, 'ScaleD',diag(SAMPLE_STD));%,'XTol',1e-6,'FunTol',1e-6,'MaxIter',1000);%,'Lambda',[]);%,'MaxFunEvals',200);
%[p ssq cnt nfj] = LMFnlsq2(@(p) nerror_LMF_Chi_g(p,b.',w,TIME,SAMPLE_STD),p0);%, 'ScaleD',diag(SAMPLE_STD));%,'XTol',1e-6,'FunTol',1e-6,'MaxIter',1000);%,'Lambda',[]);%,'MaxFunEvals',200);
%[p ssq cnt] = LMFsolve(@(p) nerror_LMF_g(p,b.',w,TIME),p0);
%[p ssq cnt nfj] = LMFnlsq(@(p) robustError_LMF_g(p,b.',w,TIME),p0);%,'Display','off','XTol',[],'FunTol',[],'MaxIter',1000,'Lambda',[]);%,'MaxFunEvals',200);



%Powell's
%[p,Ot,nS]=powell(@(p) nerror_Powell_g(p,b.',w,TIME),p0,0,0,[],[],-1,1e-4,50); %Coggins
%[p,Ot,nS]=powell(@(p) nerror_Powell_Chi_g(p,b.',w,TIME,SAMPLE_STD),p0,0,0,[],[],-1,1e-4,50); %Coggins
%[p,Ot,nS]=powell(@(p) robustError_Powell_g(p,b.',w,TIME),p0,0,0,[],[],-1,1e-4,50); %Coggins
%[p,Ot,nS]=powell(@(p) nerror_Powell_g(p,b.',w,TIME),p0,0,1,[],[],-1,1e-4,50);% Golden section
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold off
%EXITFLAG
%FVAL

%EXXXXXX = EXITFLAG
%OUTTTTT = OUTPUT
%RESSSSS = RESNORM
%ssssq = ssq

%Optimum parameters
To = p(1);
%dT = p(2)
ts = p(2);
tm = p(3);
% w = p(5);
% k = p(5);
Ta = p(4);
w1 = p(5);
w2 = p(6);

if Ta~=0
    % k = w/pi*(atan(pi/w*(ts-tm))-dT/Ta*asin(pi/w*(ts-tm)));

    %Calculation of the exponential damping factor(initialisation)
    %k = w/pi*(atan(pi/w*(ts-tm))-dT/Ta*asin(pi/w*(ts-tm)));  %pi/w*(ts-tm) must be in [-1,1]
    k = w2/pi*(1/tan(pi/w2*(ts-tm)));%-dT/Ta*(1/sin(pi/w*(ts-tm))));  %pi/w*(ts-tm) must be in [-1,1]

    %The cosine model function
    %c1 = (To+Ta*cos(pi/w*(t-tm))).*(t<ts)+((To+dT)+(Ta*cos(pi/w*(ts-tm))-dT)*exp(-(t-ts)/k)).*(t>=ts);
    c1_1 = To+Ta*cos(pi/w1*(t-tm));%.*(t<ts)
    c1_1(t>=tm) = 0;
    c1_2 = To+Ta*cos(pi/w2*(t-tm));%.*(t<ts)
    c1_2(t>=ts) = 0;
    c1_2(t<tm) = 0;
    c1_3 = To + Ta*cos(pi/w2*(ts-tm))*exp(-(t-ts)/k);%.*(t>=ts);
    c1_3(t<ts) = 0;
    c1 = c1_1 + c1_2 + c1_3;
else
    sk = sign(1/sin(pi/w2*(ts-tm)));
    sk(sk==0) = eps;
    k = -Inf*sk;
    
    c1 = To*ones(size(t));
end
%kkk = k

%dTTTT = c1(1) - c1(end)

%length(find(isnan(c1)==1))

% if EXITFLAG==0
%     figure;
%     plot(TIME,b);
%     hold on
%     plot(TIME,c1,'r');
%     hold off
%     figure
%     %FVAL
%     %EXITFLAG
%     %OUTPUT
% end

% figure
%plot(TEST_CYCLE_TIME,c1,'r') 
% xlabel('Time stamp')
% ylabel('Brightness temperature (K)')
% hold on 
% plot(t(find(mc==0)),b(find(mc==0)),'bx')
% if sum(mc)>0
%     plot(t(find(md)),b(find(md)),'rs')
% end
%   legend('Interpolated curve','Valid samples','Missing Samples')
% hold off
% 
% % Calculate the fit error to all data.
% Err = c1.'-b;
% mse0 = mean(Err.^2);                %Mean Square Error for all samples
% std0 = std(Err(find(mc==0)).^2);    %Standard deviation
% 
% % Calculate the fit error to low integrity data  in other words where clouds present (i.e. where mc = 1)
% mse1 = mean((Err.^2).*mc);          %Mean of Square Errors
% std1 = std(Err(find(mc==1)).^2);    %Standard deviation
% [mse0, std0, mse1, std1]
% 
% save c1 c1
% 
% v = a;
% v1 = v(1:96);
% v2 = v(97:192);
% v3 = v(193:288);
% v4 = v(289:384);
% v5 = v(385:480);
% v6 = v(481:576);
% Q = var([v1 v2 v3 v4 v5 v6],'',2);
% save Q Q
 

%tDDD = t












