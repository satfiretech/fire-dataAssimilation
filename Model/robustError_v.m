function mse = nerror_v(b,TIME,p)

% nerror_g returns the mean square error for the cosine model for
% fgiven parameters of the cosine model function.
% p keeps the 6 parameters of the cosine model, w is daylength, TIME is the timestamp of the cycle
%The function is used with diurnal_g


global FLAG
global ROB_SIGMA


t = TIME;

To = p(1);
%dT = p(2);
ts = p(2);
tm = p(3);
%w = p(5);
%k = p(5);
Ta = p(4);
w1 = p(5);
w2 = p(6);



%Calculation of the exponential damping factor(initialisation)
%k = w/pi*(atan(pi/w*(ts-tm))-dT/Ta*asin(pi/w*(ts-tm)));

%if dT~=0 && Ta~=0
% if Ta~=0
%     k = w/pi*((1/tan(pi/w*(ts-tm)))-dT/Ta*(1/sin(pi/w*(ts-tm))));  %pi/w*(ts-tm) must be in [-1,1]
% else
%     k = w/pi*(1/tan(pi/w*(ts-tm)));
% end



% if Ta==0
%     Ta = Ta + eps;  %pi/w*(ts-tm) must be in [-1,1]
% end
%Ta(Ta==0) = eps;
if Ta~=0
    k = w2/pi*(1/tan(pi/w2*(ts-tm)));

    % if Ta==0
    %     if dT==0
    %         k = w/pi*(1/tan(pi/w*(ts-tm)));
    %     else
    %         Ta = dT;
    %     end
    % else
    %     k = w/pi*((1/tan(pi/w*(ts-tm)))-dT/Ta*(1/sin(pi/w*(ts-tm))));  %pi/w*(ts-tm) must be in [-1,1]
    % end

    %c = (To+Ta*cos(pi/w*(t-tm))).*(t<ts)+((To+dT)+(Ta*cos(pi/w*(ts-tm))-dT)*exp(-(t-ts)/k)).*(t>=ts);

    c1 = To+Ta*cos(pi/w1*(t-tm));%.*(t<ts)
    c1(t>=tm)=0;
    c2 = To+Ta*cos(pi/w2*(t-tm));%.*(t<ts)
    c2(t>=ts) = 0;
    c2(t<tm) = 0;
    c3 = To + Ta*cos(pi/w2*(ts-tm))*exp(-(t-ts)/k);%.*(t>=ts)
    c3(t<ts)=0;
    c = c1 + c2 + c3;
else
    c = To*ones(size(t));
end

%save TIME TIME
%save b b
%Include the case of missing data
diff = c-b;
%diff(find(isnan(diff)==1)) = Inf;
% length(find(isnan(c)==1))
% length(find(isnan(b)==1))
%mse = mean(diff.^2);
%mse = sum(diff.^2)/length(diff);
diff(isnan(diff)==1) =[];


if FLAG==0
    if sum(isinf(diff)==1)>0
        ROB_SIGMA = 1.14601*mad(diff,1);
    else
        ROB_SIGMA = std(diff);
    end
    %sg1 = mad(diff,1); %(This can also be used)
    %sg1 = mad(diff,0); %(This can also be acceptable)
    FLAG = FLAG + 1;
end
sg = 3*ROB_SIGMA;

NUMERATOR = diff.^2;
DENOMINATOR = sg.^2 + diff.^2;
QUOTIENT = NUMERATOR./DENOMINATOR;
QUOTIENT((isinf(NUMERATOR) & isinf(DENOMINATOR)) ==1) = 1;
mse = sum(QUOTIENT); %Robust error (4)


%mse = sum(diff.^2)/length(diff);
%mse = sum(diff.^2); %sum of square errors (1)
%mse = sum(abs(diff)); %Sum of absolute errors (2)
%mse = sum(log(1 + 1/2*(diff.^2))) %Lorentzian or Cauchy error function (3)


% load zeropp
% pp = pp + 1;
% save zeropp pp
% %save zeropp pp
% 
% plot(pp,mse,'*')

% if (length(find(isnan(mse)==1))~=0)
%     mse
%     To
%     dT
%     ts
%     tm
%     w
%     k
%     Ta
% end




% close all
% CCC = [];
% for k=0.001:0.001:1
% To = 280.3122;
% dT = 6.1866e-004;
% ts = 17.0708;
% tm = 11.8962;
% Ta = 20.9077;
% w = 10.3632;
% %k = 3;
% c = (To+Ta*cos(pi/w*(t-tm))).*(t<ts)+((To+dT)+(Ta*cos(pi/w*(ts-tm))-dT)*exp(-(t-ts)/k)).*(t>=ts);
% %c = (Ta*cos(pi/w*(ts-tm))-dT)*exp(-(t-ts)/k).*(t>=ts);
% %c = (Ta*cos(pi/w*(ts-tm))-dT).*(t>=ts); %*exp(-(t-ts)/k)
% % c1 = exp(-(t-ts)/k);%.*(t>=ts);
% % c = (-(t-ts)/k);%.*(t>=ts);
% % KEEP_C = [c1(t<=tm);c(t<=tm)];
% %plot(t(t>=ts),bP(t>=ts))
% %hold on
% %plot(t(t>=ts),c(t>=ts),'r')
% CC = c(t>=ts);
% CCC = [CCC CC(1) - CC(end)];
% %ppp = (t>=ts);
% end
% plot(0.001:0.001:1,CCC)
% %grid
% %hold off
% length(find(isnan(c)==1))
