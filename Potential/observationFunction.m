function c1 = observationFunction(X,Z,t)
%X: the state
%Z: observed value
%t: time

% To = zeros(NUMBER_ROW,NUMBER_COLUMN,Ns); %TESTINGCYCLE_DATA(jN,iN).To(n(jN,iN));
% Ta = zeros(NUMBER_ROW,NUMBER_COLUMN,Ns); %TESTINGCYCLE_DATA(jN,iN).Ta(n(jN,iN));
% tm = zeros(NUMBER_ROW,NUMBER_COLUMN,Ns); %TESTINGCYCLE_DATA(jN,iN).tm(n(jN,iN));
% ts = zeros(NUMBER_ROW,NUMBER_COLUMN,Ns); %TESTINGCYCLE_DATA(jN,iN).ts(n(jN,iN));
% w1 = zeros(NUMBER_ROW,NUMBER_COLUMN,Ns); %TESTINGCYCLE_DATA(jN,iN).w(n(jN,iN));
% w2 = zeros(NUMBER_ROW,NUMBER_COLUMN,Ns); %TESTINGCYCLE_DATA(jN,iN).w2(n(jN,iN));
% c= zeros(NUMBER_ROW,NUMBER_COLUMN,Ns); %TESTINGCYCLE_DATA(jN,iN).w2(n(jN,iN));
%c = ttsr - tsr

%Z:z_extended
%inn = z - prediction(:,:,timestamp); %Inn is the residual

t = extend(t,size(Z,3));


To = reshape(X(1,:,:),size(Z));
Ta = reshape(X(2,:,:),size(Z));
tm = reshape(X(3,:,:),size(Z));
ts = reshape(X(4,:,:),size(Z));
w1 = reshape(X(5,:,:),size(Z));
w2 = reshape(X(6,:,:),size(Z));
%c= reshape(X(7,:,:),size(Z));
%c = ttsr - tsr

Keep_t_Ta=zeros(size(Ta));
Keep_t_Ta(find(Ta~=0)) = 1;




k = zeros(size(To));
k(Keep_t_Ta==1) = w2(Keep_t_Ta==1)/pi.*(1./tan(pi./w2(Keep_t_Ta==1).*(ts(Keep_t_Ta==1)-tm(Keep_t_Ta==1))));%-dT/Ta*(1/sin(pi/w*(ts-tm))));  %pi/w*(ts-tm) must be in [-1,1]
sk(Keep_t_Ta==0) = sign(1./sin(pi./w2(Keep_t_Ta==0).*(ts(Keep_t_Ta==0)-tm(Keep_t_Ta==0))));
sk(sk==0) = eps;
k(Keep_t_Ta==0) = -Inf*sk(Keep_t_Ta==0);



%keep_t_ts1 = (t <= tm)*4;
%keep_t_ts3 = ((t > tm) & (t <= (ts-15/60)))*2;
%keep_t_ts = keep_t_ts1 + keep_t_ts3; %0:Last condition, 1:Middle condition, 2:First condition
%[]()[]
%[)[)[]


c1_1 = zeros(size(To));
c1_2 = zeros(size(To));
c1_3 = zeros(size(To));
c1 = zeros(size(To));



c1_1(Keep_t_Ta==1) = To(Keep_t_Ta==1)+Ta(Keep_t_Ta==1).*cos(pi./w1(Keep_t_Ta==1).*(t(Keep_t_Ta==1)-tm(Keep_t_Ta==1)));%.*(t<ts)
c1_1(t>=tm) = 0;
c1_2(Keep_t_Ta==1) = To(Keep_t_Ta==1)+Ta(Keep_t_Ta==1).*cos(pi./w2(Keep_t_Ta==1).*(t(Keep_t_Ta==1)-tm(Keep_t_Ta==1)));%.*(t<ts)
c1_2(t>=ts) = 0;
c1_2(t<tm) = 0;

%c1_3 = To(Keep_t_Ta==1)+Ta(Keep_t_Ta==1).*cos(pi./w2(Keep_t_Ta==1).*(ts(Keep_t_Ta==1)-tm(Keep_t_Ta==1))).*exp(-(t(Keep_t_Ta==1)-ts(Keep_t_Ta==1))./k(Keep_t_Ta==1));%.*(t>=ts);



CCC = exp(-(t(Keep_t_Ta==1)-ts(Keep_t_Ta==1))./k(Keep_t_Ta==1));%.*(t>=ts)
% if sum(isinf(CCC))>0
%     www2 = w2(isinf(CCC))
%     tsss = ts(isinf(CCC))
%     tmmm = tm(isinf(CCC))
% end
CCC(isinf(CCC)) = 1/eps;
c1_3(Keep_t_Ta==1) = To(Keep_t_Ta==1)+Ta(Keep_t_Ta==1).*cos(pi./w2(Keep_t_Ta==1).*(ts(Keep_t_Ta==1)-tm(Keep_t_Ta==1))).*CCC;




c1_3(t<ts) = 0;
c1(Keep_t_Ta==1) = c1_1(Keep_t_Ta==1) + c1_2(Keep_t_Ta==1) + c1_3(Keep_t_Ta==1);




    
c1(Keep_t_Ta==0) = To(Keep_t_Ta==0);%.*ones(size(t));    

%[max(k(:)) min(k(:))];

%exp(-(t(Keep_t_Ta==1)-ts(Keep_t_Ta==1))./k(Keep_t_Ta==1))


% if Ta~=0
    % k = w/pi*(atan(pi/w*(ts-tm))-dT/Ta*asin(pi/w*(ts-tm)));

    %Calculation of the exponential damping factor(initialisation)
    %k = w/pi*(atan(pi/w*(ts-tm))-dT/Ta*asin(pi/w*(ts-tm)));  %pi/w*(ts-tm) must be in [-1,1]
    %k = w2/pi*(1/tan(pi/w2*(ts-tm)));%-dT/Ta*(1/sin(pi/w*(ts-tm))));  %pi/w*(ts-tm) must be in [-1,1]

    %The cosine model function
    %c1 = (To+Ta*cos(pi/w*(t-tm))).*(t<ts)+((To+dT)+(Ta*cos(pi/w*(ts-tm))-dT)*exp(-(t-ts)/k)).*(t>=ts);
%     c1_1 = To+Ta*cos(pi/w1*(t-tm));%.*(t<ts)
%     c1_1(t>=tm) = 0;
%     c1_2 = To+Ta*cos(pi/w2*(t-tm));%.*(t<ts)
%     c1_2(t>=ts) = 0;
%     c1_2(t<tm) = 0;
%     c1_3 = To + Ta*cos(pi/w2*(ts-tm))*exp(-(t-ts)/k);%.*(t>=ts);
%     c1_3(t<ts) = 0;
%     c1 = c1_1 + c1_2 + c1_3;
% else
%    sk = sign(1/sin(pi/w2*(ts-tm)));
%    sk(sk==0) = eps;
%    k = -Inf*sk;
    
%     c1 = To*ones(size(t));
% end
