function [KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual y cycle_out_scope cycle_reported xs Gf Keep_Q2 EnsPX_ prediction predictedObservation t FEATURE1 y_predictedKeep] = stdxtkalmanDECONV10(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,DECONV,restored,Keep_Q2,NUMBER_ENSEMBLE_MEMBERS, EnsPX_, prediction, predictedObservation, threshold, FEATURE1, RESIDUAL, y_predictedKeep)
%function [KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual y cycle_out_scope cycle_reported xs Gf Keep_Q2 EnsPX_ prediction predictedObservation t] = stdxtkalmanDECONV10(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,DECONV,restored,Keep_Q2,NUMBER_ENSEMBLE_MEMBERS, EnsPX_, prediction, predictedObservation)
%function [x_ v4 t] = stdxtkalmanDECONV7(TESTINGCYCLE_DATA, timestamp,n,x_)
%function [KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual y cycle_out_scope cycle_reported xs Gf Q1 Kn Q2] = stdxtkalmanDECONV6(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,DECONV,restored)

%THIS FUNCTION PREDICTS CURRENT SAMPLE GIVEN PREVIOUS SAMPLE

KeepAlpha = [];
%%%
%%%

M = 6; %Dimension of the state variable
N = 2; %Dimension of the observation

%%%%%%%%%%%%%%%

% savCn1 = [];
%fire_detect = [];
%alpha = [];
%stResidual = [];
cycle_reported = [];

% time_ = [];

%%%
%%%


[NUMBER_ROW NUMBER_COLUMN] = size(TESTINGCYCLE_DATA);


v4 = zeros(NUMBER_ROW,NUMBER_COLUMN);
t = zeros(NUMBER_ROW,NUMBER_COLUMN);
w1 = zeros(NUMBER_ROW,NUMBER_COLUMN);
w2 = zeros(NUMBER_ROW,NUMBER_COLUMN);
To = zeros(NUMBER_ROW,NUMBER_COLUMN);
Ta = zeros(NUMBER_ROW,NUMBER_COLUMN);
tm = zeros(NUMBER_ROW,NUMBER_COLUMN);
ts = zeros(NUMBER_ROW,NUMBER_COLUMN);
%dT = zeros(NUMBER_ROW,NUMBER_COLUMN);
k = zeros(NUMBER_ROW,NUMBER_COLUMN);
%Q = zeros(NUMBER_ROW,NUMBER_COLUMN);
R1 = zeros(NUMBER_ROW,NUMBER_COLUMN);
cycle_out_scope = zeros(NUMBER_ROW,NUMBER_COLUMN);
c = zeros(NUMBER_ROW,NUMBER_COLUMN);

for jN = 1:NUMBER_ROW
    for iN = 1:NUMBER_COLUMN
        v4(jN,iN) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_DATA(n(jN,iN));
        t(jN,iN) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME(n(jN,iN));
        w1(jN,iN) = TESTINGCYCLE_DATA(jN,iN).w(n(jN,iN));
        w2(jN,iN) = TESTINGCYCLE_DATA(jN,iN).w2(n(jN,iN));
        To(jN,iN) = TESTINGCYCLE_DATA(jN,iN).To(n(jN,iN));
        Ta(jN,iN) = TESTINGCYCLE_DATA(jN,iN).Ta(n(jN,iN));
        tm(jN,iN) = TESTINGCYCLE_DATA(jN,iN).tm(n(jN,iN));
        ts(jN,iN) = TESTINGCYCLE_DATA(jN,iN).ts(n(jN,iN));
        %dT(jN,iN) = TESTINGCYCLE_DATA(jN,iN).dT(n(jN,iN));
        k(jN,iN) = TESTINGCYCLE_DATA(jN,iN).k(n(jN,iN));
        
        %Q(jN,iN) = TESTINGCYCLE_DATA(jN,iN).Q(n(jN,iN));
        
        R1(jN,iN) = TESTINGCYCLE_DATA(jN,iN).R1(n(jN,iN)); 
        %Enter variance in tHorizon - tThermalSunrise
        %R1c
        R1c(jN,iN) = TESTINGCYCLE_DATA(jN,iN).R1c(n(jN,iN));
        cycle_out_scope(jN,iN) = TESTINGCYCLE_DATA(jN,iN).TREATED(n(jN,iN));
        
        tHorizon(jN,iN) = TESTINGCYCLE_DATA(jN,iN).tmorningHorizon(n(jN,iN));
        tThermalSunrise(jN,iN) = TESTINGCYCLE_DATA(jN,iN).tSunriseDay(n(jN,iN));
        
        tSolarNoon(jN,iN) = TESTINGCYCLE_DATA(jN,iN).tSolarNoon(n(jN,iN));
        tSunset(jN,iN) = TESTINGCYCLE_DATA(jN,iN).tSunsetDay(n(jN,iN));
        
    end
end
tNoon = reshape(tSolarNoon,1,size(tSolarNoon,1)*size(tSolarNoon,2));
tSet = reshape(tSunset,1,size(tSunset,1)*size(tSunset,2));
c = tThermalSunrise - tHorizon; 
%R1c = R1;


%CHANGE HERE for Q, it must be trained
for iN = 1:NUMBER_COLUMN
    for jN = 1:NUMBER_ROW
        %Q(1:6,1+((jN + NUMBER_ROW*(iN-1))-1)*6:6+((jN + NUMBER_ROW*(iN-1))-1)*6) = TESTINGCYCLE_DATA(jN,iN).Q(n(jN,iN))/10*eye(6); %1
        %Q(:,(iN-1)+1:(iN-1)+7)
        Q(1:6,1+((jN + NUMBER_ROW*(iN-1))-1)*6:6+((jN + NUMBER_ROW*(iN-1))-1)*6) = TESTINGCYCLE_DATA(jN,iN).CovParameter;
        EIG = eig(TESTINGCYCLE_DATA(jN,iN).CovParameter);
        MULTIPLIER(1:6,jN + NUMBER_ROW*(iN-1)) = EIG(end)*ones(6,1);
        %TRACE = trace(TESTINGCYCLE_DATA(jN,iN).CovParameter);
        %MULTIPLIER(1:6,jN + NUMBER_ROW*(iN-1)) = TRACE*ones(6,1);
    end
end  %7 x 7 x Pixels

%Fix R1 to be 2 x 2 x Pixels

Ne = NUMBER_ENSEMBLE_MEMBERS;
%%%
%%%

if DECONV==1
    y = restored;
else
    y = v4; %I can optimize the code by not reading v4 in case of DECONV = 1
    %[y = [1 x Number of rows x Number of columns]
    %[c = [1 x Number of rows x Number of columns]
end

Q1 = Q; %Process noise covariance
% y(:,:,timestamp) = v4;
% Q1(:,:,timestamp) = Q;

yo = [reshape(y,1,size(y,1)*size(y,2));
    reshape(c,1,size(c,1)*size(c,2))];

%%%
%%%
 
xs = zeros(6,NUMBER_ROW*NUMBER_COLUMN,Ne); %xf is filtered samples, it will keep all filtered distribution, each distribution on a each pixels
process_noise = normalizedState(Q1,randn(size(xs))); %[Number of variables x Number of pixels]
%process_noise = normalizedStateA(Q1,randn(size(xs))); %[Number of variables x Number of pixels] --- Whiten the random number first and then derive random number of a certain covariance matrix

%Check why is w1 not w2 also
if timestamp==1

    xi = [reshape(To,1,size(To,1)*size(To,2));
        reshape(Ta,1,size(Ta,1)*size(Ta,2));
        reshape(tm,1,size(tm,1)*size(tm,2));
        reshape(ts,1,size(ts,1)*size(ts,2));
        reshape(w1,1,size(w1,1)*size(w1,2));
        reshape(w2,1,size(w2,1)*size(w2,2))];
 %       reshape(c,1,size(c,1)*size(c,2))];

       
    %x_(1:size(t,1),1:size(t,2),timestamp) = To+Ta.*cos(pi./w1.*(t-tm)); %xs; %TO CHECK
    x_= extend(xi,Ne) + process_noise;

    
    %Kn_(1:size(t,1),1:size(t,2),timestamp) = Q; %10;%xcorr
    EnsPX_ = x_ - extend(mean(x_,3),Ne);
    %Kn_(1:size(t,1),1:size(t,2),timestamp) = calculateForecastErrorCovariance(EnsPX_);
    Kn_ = calculateForecastErrorCovariance(EnsPX_);
    prediction(:,:,timestamp) = mean(x_,3);
    
    %prediction(:,:,1)
    %mean(process_noise,3)
    
    predictedObservation(:,:,timestamp) = observationFunction(prediction(:,:,timestamp),y,t);
end

%C = [];%deltaCnl/deltax at x = xn_
%F;

%Enter the algorithm with a prediction of current time
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Q2 = R1; %KEEP IT AS PREDICTED RESIDUAL COVARIANCE. %TO BE REMOVED THERE ARE REPEATED IN THE THRESHOLD SETTING Q = R
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % S = C*Kn_(:,:,timestamp)*C'+Q2; %Innovation or Residual covariance
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Gf = Kn_(:,:,timestamp).*C'.*(S).^(-1); %Kalman gain %THIS CAN BE REMOVED AS IS CALCULATED AFTER DETECTION


%Cn1 = x_(:,:,timestamp);


% savCn1(:,:,timestamp) = Cn1; %Save predicted
% estimates~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~here

% if DECONV==1
%     alpha = y - prediction(:,:,timestamp);
% else
%alpha = y - Cn1; %Innovation vector (prediction residual)
% end
Q2 = R1; %KEEP IT AS PREDICTED RESIDUAL COVARIANCE. %TO BE REMOVED THERE ARE REPEATED IN THE THRESHOLD SETTING Q = R
%Q2c = R1c;

% Q2x = reshape(R1,1,size(R1,1)*size(R1,2));
% Q2c = reshape(R1c,1,size(R1c,1)*size(R1c,2));
% for i = 1:size(R1,1)*size(R1,2)
%     Q2a(1,1+(i-1)*2:2+(i-1)*2) = [Q2x(i) 0]; 
%     Q2a(2,1+(i-1)*2:2+(i-1)*2) = [0 Q2c(i)];
% end


y_extended = extend(y,Ne);
y_perturbed = y_extended + extend(sqrt(Q2),Ne).*randn(size(y_extended));
% disp('1')
y_predicted = observationFunction(x_,y_perturbed,t);
alpha = y_perturbed - y_predicted; %50 x 50 x Ne 
inn = y-reshape(mean(y_predicted,3),size(y,1),size(y,2)); %then, this gives 50 x 50 
%No need of reshape at inn
%YYYYY = mean(y_predicted,3)
% size(alpha)
% size(inn)
% size(y_perturbed)
% size(y_predicted)


%1 x 2500 x Ne for a 50 x 50 images

c_extended = extend(c,Ne);
c_perturbed = c_extended + extend(sqrt(R1c),Ne).*randn(size(c_extended));
c_predicted = thermalsunriseFunction(x_,c_perturbed,t,extend(tHorizon,Ne));
alphac = c_perturbed - c_predicted; %50 x 50 x Ne

% size(alphac)
% size(c_perturbed)
% size(c_predicted)


% if sum(isnan(x_))>0
%     disp('x_ has Nan')
% end
% 
% 
% if sum(isnan(y_predicted))>0
%     disp('y_predicted has Nan')
% end

% x_ = x_ + process_noise;
% y_predicted = observationFunction(x_,y_perturbed,t);
% c_predicted = thermalsunriseFunction(x_,c_perturbed,t,tHorizon);



%KeepAlpha(1:size(t,1),1:size(t,2),timestamp) = alpha;


% KeepAlpha(1:size(t,1),1:size(t,2),timestamp) = alpha;

% disp('2')
HX = y_predicted - extend(mean(y_predicted,3),Ne);
HXc = c_predicted - extend(mean(c_predicted,3),Ne);

% [mean(y_predicted,3) min(HX(:)) max(HX(:)) ]
% [mean(c_predicted,3) min(HXc(:)) max(HXc(:))]

HXa = zeros(2,size(HX,1)*size(HX,2),size(HX,3));
HXa(1,:,:) = reshape(HX,1,size(HX,1)*size(HX,2),size(HX,3)); %2 x Pixels x Ne
HXa(2,:,:) = reshape(HXc,1,size(HXc,1)*size(HXc,2),size(HXc,3));

HPH = [];
%size(squeeze(HXa(:,i,:)))
%size(HPH)
for i=1:size(HXa,2) %Number of pixels
    %HPH(1:2,1+(i-1)*2:2+(i-1)*2) = 
    HPH = [HPH squeeze(HXa(:,i,:))*(squeeze(HXa(:,i,:))).'/(Ne-1)]; %2 x (2 . Pixels)
end
%HPH = sum(HX.*HX,3)/(Ne-1);
%S = C.*Kn_(1:size(t,1),1:size(t,2),timestamp).*C'+Q2; %Innovation or Residual covariance
%Gf = Kn_(:,:,timestamp).*C'.*(S).^(-1); %Kalman gain %THIS CAN BE REMOVED AS IS CALCULATED AFTER DETECTION
%S = reshape(HPH,size(Q2)) + Q2;
% S = HPH + Q2a;


%[HPH inv(S)]
%[x_(:,:,find(y_predicted==min(y_predicted(:)))) x_(:,:,find(y_predicted==max(y_predicted(:))))]
%Keep_S(1:size(t,1),1:size(t,2),timestamp) = S; %2 x (2 . Pixels) x timestamp
% Keep_S(1:2,1:size(t,1)*size(t,2)*2,timestamp) = S; %2 x (2 . Pixels) x timestamp
% Keep_Q2(1:size(t,1),1:size(t,2),timestamp) = Q2; %Keep the measurement covariance
%Keep_Q2(1:2,1:size(t,1)*size(t,2)*2,timestamp) = Q2a; %Keep the measurement covariance


% if sum(isnan(y_predicted))>0
%     disp('y_predicted2 has Nan')
% end
% 
% if sum(isnan(mean(y_predicted,3)))>0
%     disp('y_mean2 has Nan')
% end

% if sum(isnan(HX))>0
%     mean(y_predicted,3)
%     disp('HX has Nan')
% end


%To consider NaN = Cycle with no training
%I can also keep the x_ (KEEP_x_)before restoration so that during contextual, those
%points will kept as NaN
% xx_ = x_;
% KK_ = Kn_;


% if timestamp ==300
%     AA = Kn_(:,:,timestamp);
%     length(find(isnan(AA)==1))
%     length(find(isnan(S)==1))
%     length(find(isnan(Q2)==1))
% end

% BB = Kn_(:,:,timestamp);
% %CC = alpha;
% %DD = x_(:,:,timestamp);
% EE = Q2;
% %FF = S;

% %[length(find(isnan(Cn1(:))==1)) length(find(isnan(Q2)==1))]

%x_(:,:,timestamp)

%No need to restore missing data for the residuals
% if length(find(isnan(alpha)==1))~=0
%     windowSize_row = size(alpha,1);
%     windowSize_column = size(alpha,2);
%     window_start_row = 1;
%     window_start_column = 1;
%     %     window_centre_row = ceil(windowSize_row/2); %Case the window size is an even number? What will happen?
%     %     window_centre_column = ceil(windowSize_column/2); %Case the window size is an even number? What will happen?
% 
%     alpha = restoreMissingSamples10(alpha,timestamp,windowSize_row,windowSize_column,window_start_row, window_start_column,KeepAlpha);
%     %alpha=LaplacianInterpolation(alpha);
%     Cn1 = y - alpha;
%     x_(:,:,timestamp) = Cn1;
% end


%No need to restore missing value for the forecast error covariance
% tempKn_(1:size(Kn_,1),1:size(Kn_,2)) = Kn_(:,:,timestamp);
% if  length(find(isnan(tempKn_)==1))~=0
%     windowSize_row = size(tempKn_,1);
%     windowSize_column = size(tempKn_,2);
%     window_start_row = 1;
%     window_start_column = 1;
%     %     window_centre_row = ceil(windowSize_row/2); %Case the window size is an even number? What will happen?
%     %     window_centre_column = ceil(windowSize_column/2); %Case the window size is an even number? What will happen?
% 
%     SRKn_ = restoreMissingSamples10(sqrt(Kn_(:,:,timestamp)),timestamp,windowSize_row,windowSize_column,window_start_row, window_start_column,sqrt(Kn_));
%     SRKn_(SRKn_<0) = 0;
%     Kn_(:,:,timestamp) = SRKn_.^2;
% end


% if  length(find(isnan(S)==1))~=0
%     windowSize_row = size(S,1);
%     windowSize_column = size(S,2);
%     window_centre_row = ceil(windowSize_row/2); %Case the window size is an even number? What will happen?
%     window_centre_column = ceil(windowSize_column/2); %Case the window size is an even number? What will happen?
%     SRS = restoreMissingSamples6(sqrt(S),timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,sqrt(Keep_S));
%     SRS(SRS<0) = 0;
%     S = SRS.^2;
%     %Kn_(:,:,timestamp) = restoreMissingSamples6(Kn_(:,:,timestamp),timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,Kn_);
% end


%alpha = restoreMissingSamples6(alpha,timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,KeepAlpha);
%alpha=LaplacianInterpolation(alpha);
%Cn1 = y - alpha;
%x_(:,:,timestamp) = Cn1;


if  length(find(isnan(Q2)==1))~=0
    windowSize_row = size(Q2,1);
    windowSize_column = size(Q2,2);
    window_start_row = 1;
    window_start_column = 1;
    %     window_centre_row = ceil(windowSize_row/2); %Case the window size is an even number? What will happen?
    %     window_centre_column = ceil(windowSize_column/2); %Case the window size is an even number? What will happen?

    %Kn_(:,:,timestamp) = restoreMissingSamples6(Kn_(:,:,timestamp),timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,Kn_);

    %Kn_(:,:,timestamp) = LaplacianInterpolation(Kn_(:,:,timestamp));
    %Two methods, one using Kn_ and another one using S
    %Kn_(:,:,timestamp) = restoreMissingSamples6(Kn_(:,:,timestamp),timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,Kn_);

%     SRQ2 = restoreMissingSamples10(sqrt(Q2),timestamp,windowSize_row,windowSize_column,window_start_row, window_start_column,sqrt(Keep_Q2));
%     SRQ2(SRQ2<0) = 0;
%     Q2 = SRQ2.^2;
    %S = restoreMissingSamples6(S,timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,Keep_S);
    %S = Kn_(:,:,timestamp) + Q2;
    %Kn_(:,:,timestamp) = S - Q2; %Considering that C is a scalar equal to 1
end
%S = Kn_(:,:,timestamp) + Q2;
%Kn_(:,:,timestamp) = S  - Q2;

%Q2 = R1
Q2x = reshape(Q2,1,size(R1,1)*size(R1,2));
Q2c = reshape(R1c,1,size(R1c,1)*size(R1c,2));
for i = 1:size(R1,1)*size(R1,2)
    Q2a(1,1+(i-1)*2:2+(i-1)*2) = [Q2x(i) 0]; 
    Q2a(2,1+(i-1)*2:2+(i-1)*2) = [0 Q2c(i)];
end
S = HPH + Q2a;
% if isinf(S(1))==1
%     S([1 4]) = 1/eps;
% end
%S(isinf(S)) = 1/eps;

% if timestamp ==300
%     AA = Kn_(:,:,timestamp);
%     length(find(isnan(AA)==1))
%     length(find(isnan(S)==1))
%     length(find(isnan(Q2)==1))
% end


%if timestamp>=219 & timestamp<=222
%    cycle_out_scope
%end

% if cycle_out_scope==1
%     [xx_ x_ KK_ Kn_]
% end

%KeepAlpha(1:size(t,1),1:size(t,2),timestamp) = alpha; %Update alpha
% Keep_S(1:size(t,1),1:size(t,2),timestamp) = S;
% Keep_Q2(1:size(t,1),1:size(t,2),timestamp) = Q2; %Update measurement covariance
Keep_S(1:2,1:size(t,1)*size(t,2)*2,timestamp) = S; %2 x (2 . Pixels) x timestamp
Keep_Q2(1:2,1:size(t,1)*size(t,2)*2,timestamp) = Q2a; %Keep the measurement covariance



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     DETECTION TYPE 1
%No need to detect fire
% Tc = sqrt(S).*NUMBER_STANDARD_DEVIATION; %Tc = 1; corresponds to sqrt(S(n))*0.5
%Tc = NUMBER_STANDARD_DEVIATION;

%Tc = 100;
%fire_detect = alpha > Tc;
%No need to detect fire
% fire_detect = alpha > inn;
% KeepFireDetect(1:size(t,1),1:size(t,2),timestamp) = fire_detect;
%read_fire_detect = fire_detect;
%Gf = mod(fire_detect + 1,2); %CASE OF FIRE
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % R = R1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Q2 = R;
%S = C.*Kn_(:,:,timestamp).*C'+Q2; %This line and the following
%S = reshape(HPH,size(Q2)) + Q2;
% S(S==0) = eps;
%Gf = mod(fire_detect + 1,2).*Kn_(:,:,timestamp).*C'.*(S).^(-1); %Done two times, abuse CASE OF NO FIRE (Fire=1, no fire = 0)
PH = backgroundValueOfObservation4(EnsPX_,HXa); % 6 x (2.pixels)
%Gf = mod(fire_detect + 1,2).*PH.*(reshape(S,1,size(S,1)*size(S,2))).^(-1); %Done two times, abuse CASE OF NO FIRE (Fire=1, no fire = 0)


% Gf = PH.*(reshape(S,1,size(S,1)*size(S,2))).^(-1); %Done two times, abuse CASE OF NO FIRE (Fire=1, no fire = 0)

% if sum(isnan(PH(:)))>0
%     disp('PH has Nan')
% end
% 
% if sum(isnan(S(:)))>0
%     disp('S has Nan')
% end



%-----------
%Get fire detection result at launch of the EnKF
if timestamp==1
    if RESIDUAL == 0
        [FIREFLAG FEATURE1 y_predictedM] = startTimeFireDetect(y, Keep_Q2, x_, t, threshold, RESIDUAL);
        KeepFireDetect(:,:,timestamp) = FIREFLAG;
        y_predictedKeep(:,:,timestamp) = y_predictedM;
  
    elseif RESIDUAL == 1
        [FIREFLAG FEATURE1 y_predictedM] = startTimeFireDetect(y, Keep_Q2, x_, t, threshold, RESIDUAL);
        KeepFireDetect(:,:,timestamp) = FIREFLAG;
        y_predictedKeep(:,:,timestamp) = y_predictedM;

    elseif RESIDUAL == 2
        [FIREFLAG FEATURE1 y_predictedM] = startTimeFireDetect(y, Keep_Q2, x_, t, threshold, RESIDUAL);
        KeepFireDetect(:,:,timestamp) = FIREFLAG;
        y_predictedKeep(:,:,timestamp) = y_predictedM;
        
    else
        display('specify the feature')
    end
end
%--------------------------


KeepFireDetectx = reshape(KeepFireDetect(:,:,timestamp),1,size(KeepFireDetect,1)*size(KeepFireDetect,2));
%Another way would be to write R1(KeepFireDetect(:,:,timestamp)==0) =
%1/eps; R1c(KeepFireDetect(:,:,timestamp)==0) = 1/eps; but the problem is
%when HPH = [x inf;inf x]
Gf = [];
for i=1:size(HXa,2) %Number of pixels
    if KeepFireDetectx(i)==0  %& isinf(S(1))==0
        %Gf = [Gf PH(1:6,1+(i-1)*2:2+(i-1)*2)*inv(S(1:2,1+(i-1)*2:2+(i-1)*2))]; %6 x (2.Pixels)
        Gf = [Gf PH(1:6,1+(i-1)*2:2+(i-1)*2)*(S(1:2,1+(i-1)*2:2+(i-1)*2)\[1 0;0 1])]; %6 x (2.Pixels)
    else
        Gf = [Gf PH(1:6,1+(i-1)*2:2+(i-1)*2)*zeros(2,2)];
    end
    %TIMET = timestamp
    %[PH;inv(S)]
%    Gf
%    PH
%    inv(S)
end
%= squeeze(HXa(:,i,:)).*squeeze(HXa(:,i,:)).'/(Ne-1); %2 x (2 . Pixels)

% if sum(isnan(inv(S)))>0
%     disp('invS has Nan')
% end

% CCC = TESTINGCYCLE_DATA(jN,iN).Q(n(jN,iN))
% if sum(isnan(Gf(:)))>0
%     disp('Gf has Nan')
% end



% if sum(isnan(x_))>0
%     disp('x_ has Nan')
% end


% if sum(isnan(PH))>0
%     disp('PH has Nan')
% end
% 
% SS = 1./S;
% if sum(isnan(SS))>0
%     disp('SS has Nan')
% end







% if timestamp >=94 && timestamp <=98
%     %length(find(isnan(CC)==1))
%     AA = Kn_(:,:,timestamp);
%     AA(find(AA<0))
%     BB(find(AA<0))
%
%     %CC(find(AA<0))
%     %alpha(find(AA<0))
%
%     %DD(find(AA<0))
%
%     %EE(find(AA<0))
%     %Q2(find(AA<0))
%
%
%
%     Gf(find(AA<0))
%
% %     FF(find(AA<0))
% %     S(find(AA<0))
% end


% Case of missed samples:
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Gf(find(isnan(y)==1)) = 0; %residual (alpha) = NaN and Gf = 0 and pixel not on fire
%fire_detect(find(isnan(y)==1)) = 2:case of undecided

% if timestamp>=220 & timestamp<=223
% %     Gf
% %     Tc
% end


%     if (alpha(:,:,n)>Tc)
%         fire_detect(:,:,n) = 1;
%         Gf(:,:,n) = 0; %ANOTHER OPTION
%     else
%         fire_detect(:,:,n) = 0;
%         R = R1;
%         Q2 = R;
%         S(n) = C*Kn_(:,:,n).*C'+Q2; %This line and the following
%         Gf(n)= Kn_(:,:,n).*C'.*(S(:,:,n)).^(-1); %Done two times, abuse
% %         OK as for the subsequent prediction the observation is taken into account kn_n(small)/Kn_n(small) + R(current)
%     end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%xs = x_(:,:,timestamp) + Gf.*alpha; %Filtered  estimate while x_(n) is predicted estimate
%xs = x_ + extend(Gf,Ne).*alpha; %Filtered  estimate while x_(n) is predicted estimate
%alpha, alphac

% if sum(isnan(Gf(:)))>0
%     disp('Gf has Nan')
% end



xs = x_ + incremental4(Gf,alpha,alphac); %Filtered  estimate while x_(n) is predicted estimate
%[mean(x_,3) mean(xs,3)]
%squeeze(mean(incremental4(Gf,alpha,alphac),3))
%alpha
%alphac



% if sum(isnan(xs(:)))>0
%     disp('xs has Nan')
% end


% if sum(isnan(Gf))>0
%     disp('Gf has Nan')
% end

%else

%dC = cos(pi./w2.*(t-tm + 15/60)) - cos(pi./w1.*(t-tm));

%xs = x_(:,:,timestamp-1);

%Check these limits
% keep_t_ts1 = (t <= tm-15/60)*4;
% keep_t_ts2 = ((t > tm-15/60) & (t < tm))*3;
% keep_t_ts3 = ((t >= tm) & (t <= ts-15/60))*2;
%
% %keep_t_ts1 = (t < tm)*4;
% %keep_t_ts3 = ((t >= tm) & (t <= ts-15/60))*2;
% %keep_t_ts3 = ((t >= tm) & (t <= ts-15/60))*2;
% keep_t_ts4 = (t>(ts-15/60)) & (t<ts);
% keep_t_ts = keep_t_ts1 + keep_t_ts2 + keep_t_ts3 + keep_t_ts4; %0:Last condition, 1:Middle condition, 2:First condition




%ANOTHER RIGHT
% keep_t_ts1 = (t <= (tm-15/60))*4;
% keep_t_ts2 = ((t > tm-15/60) & (t <= tm))*3;
% keep_t_ts3 = ((t > tm) & (t <= (ts-15/60)))*2;
% keep_t_ts4 = (t>(ts-15/60)) & (t<ts);
% keep_t_ts = keep_t_ts1 + keep_t_ts2 + keep_t_ts3 + keep_t_ts4; %0:Last condition, 1:Middle condition, 2:First condition



% keep_t_ts1 = (t <= tm)*4;
% keep_t_ts3 = ((t > tm) & (t <= (ts-15/60)))*2;
% keep_t_ts = keep_t_ts1 + keep_t_ts3; %0:Last condition, 1:Middle condition, 2:First condition

%keep_t_tm = sign((t<tm)-0.5);


%%
%%



% Adj = zeros(size(t,1),size(t,2));
% %Adj(find(keep_t_ts==2)) = real(cos(pi./(w(find(keep_t_ts==2))*4)) - keep_t_tm(find(keep_t_ts==2)).*(xs(find(keep_t_ts==2))-To(find(keep_t_ts==2))).*sin(pi./(w(find(keep_t_ts==2))*4))./sqrt(Ta(find(keep_t_ts==2)).^2 - (xs(find(keep_t_ts==2))-To(find(keep_t_ts==2))).^2));
% Adj(keep_t_ts==4) = real(cos(pi./(w1(keep_t_ts==4)*4)) - (xs(keep_t_ts==4)-To(keep_t_ts==4)).*sin(pi./(w1(keep_t_ts==4)*4))./sqrt(Ta(keep_t_ts==4).^2 - (xs(keep_t_ts==4)-To(keep_t_ts==4)).^2));
% 
% Adj(keep_t_ts==2) = real(cos(pi./(w2(keep_t_ts==2)*4)) + (xs(keep_t_ts==2)-To(keep_t_ts==2)).*sin(pi./(w2(keep_t_ts==2)*4))./sqrt(Ta(keep_t_ts==2).^2 - (xs(keep_t_ts==2)-To(keep_t_ts==2)).^2));



%         if t<tm %sin((t-tm)*pi/w)<0
%             F = real(cos(pi./(w*4)) - (xs-To).*sin(pi./(w*4))./sqrt(Ta.^2 - (xs-To).^2));
%         else
%             F = real(cos(pi./(w*4)) + (xs-To).*sin(pi./(w*4))./sqrt(Ta.^2 - (xs-To).^2));
%         end


%     elseif (t>(ts-15/60)) & (t<ts)

%Adj(find(keep_t_ts==1)) = real((cos(pi./w(find(keep_t_ts==1)).*(ts(find(keep_t_ts==1))-t(find(keep_t_ts==1)))) + (xs(find(keep_t_ts==1))-To(find(keep_t_ts==1))).*sin(pi./w(find(keep_t_ts==1)).*(ts(find(keep_t_ts==1))-t(find(keep_t_ts==1))))./sqrt(Ta(find(keep_t_ts==1)).^2 - (xs(find(keep_t_ts==1))-To(find(keep_t_ts==1))).^2)).*exp(-(t(find(keep_t_ts==1))+15/60-ts(find(keep_t_ts==1)))./k(find(keep_t_ts==1))));
%Adj(keep_t_ts==1) = real((cos(pi./w(keep_t_ts==1).*(ts(keep_t_ts==1)-t(keep_t_ts==1))) + (xs(keep_t_ts==1)-To(keep_t_ts==1)).*sin(pi./w(keep_t_ts==1).*(ts(keep_t_ts==1)-t(keep_t_ts==1)))./sqrt(Ta(keep_t_ts==1).^2 - (xs(keep_t_ts==1)-To(keep_t_ts==1)).^2)).*exp(-(t(keep_t_ts==1)+15/60-ts(keep_t_ts==1))./k(keep_t_ts==1)));

%     else
%Adj(find(keep_t_ts==0)) = exp(-1/4./k(find(keep_t_ts==0)));
% Adj(keep_t_ts==0) = exp(-1/4./k(keep_t_ts==0));
% %     end
% 
% F = Adj;

%Pr = zeros(size(t,1),size(t,2));
%     if t<=(ts-15/60)  %Predict for the next sample
%         if t<tm  %sin((t-tm)*pi/w)<0
%Pr(find(keep_t_ts==2)) = real(To(find(keep_t_ts==2))+(xs(find(keep_t_ts==2)) - To(find(keep_t_ts==2))).*cos(pi./(w(find(keep_t_ts==2))*4)) + keep_t_tm(find(keep_t_ts==2)).*sqrt(Ta(find(keep_t_ts==2)).^2 - (xs(find(keep_t_ts==2))-To(find(keep_t_ts==2))).^2) .*sin(pi./(w(find(keep_t_ts==2))*4)));





%%
%%



% % % % % % Pr = zeros(size(t,1),size(t,2));
% % % % % % %     if t<=(ts-15/60)  %Predict for the next sample
% % % % % % %         if t<tm  %sin((t-tm)*pi/w)<0
% % % % % % %Pr(find(keep_t_ts==2)) = real(To(find(keep_t_ts==2))+(xs(find(keep_t_ts==2)) - To(find(keep_t_ts==2))).*cos(pi./(w(find(keep_t_ts==2))*4)) + keep_t_tm(find(keep_t_ts==2)).*sqrt(Ta(find(keep_t_ts==2)).^2 - (xs(find(keep_t_ts==2))-To(find(keep_t_ts==2))).^2) .*sin(pi./(w(find(keep_t_ts==2))*4)));
% % % % % % Pr(keep_t_ts==4) = real(To(keep_t_ts==4)+(xs(keep_t_ts==4) - To(keep_t_ts==4)).*cos(pi./(w1(keep_t_ts==4)*4)) + sqrt(Ta(keep_t_ts==4).^2 - (xs(keep_t_ts==4)-To(keep_t_ts==4)).^2) .*sin(pi./(w1(keep_t_ts==4)*4)));
% % % % % % %x_(:,:,timestamp+1) = Pr;
% % % % % % %         else
% % % % % % %             x_(:,:,n+1) = real(To+(xs - To).*cos(pi./(w*4)) - sqrt(Ta.^2 - (xs-To).^2) .*sin(pi./(w*4)));
% % % % % % %         end
% % % % % % %     elseif (t> (ts-15/60)) & (t<ts)
% % % % % % %Pr(find(keep_t_ts==1)) = real((To(find(keep_t_ts==1)) + dT(find(keep_t_ts==1))) + ((xs(find(keep_t_ts==1))-To(find(keep_t_ts==1))).*cos(pi./w(find(keep_t_ts==1)).*(ts(find(keep_t_ts==1))-t(find(keep_t_ts==1)))) -sqrt(Ta(find(keep_t_ts==1)).^2 - (xs(find(keep_t_ts==1))-To(find(keep_t_ts==1))).^2).*sin(pi./w(find(keep_t_ts==1)).*(ts(find(keep_t_ts==1))-t(find(keep_t_ts==1))))- dT(find(keep_t_ts==1))) .*exp(-(t(find(keep_t_ts==1)) + 15/60 -ts(find(keep_t_ts==1)))./k(find(keep_t_ts==1))));
% % % % % % 
% % % % % % Pr(keep_t_ts==3) = real(xs(keep_t_ts==3) + Ta(keep_t_ts==3).*dC(keep_t_ts==3));
% % % % % % %Pr(keep_t_ts==3) = real(To(keep_t_ts==3)+(xs(keep_t_ts==3) - To(keep_t_ts==3)).*cos(pi./(w1(keep_t_ts==3)*4)) + sqrt(Ta(keep_t_ts==3).^2 - (xs(keep_t_ts==3)-To(keep_t_ts==3)).^2) .*sin(pi./(w1(keep_t_ts==3)*4)) + Ta.*dC(keep_t_ts==3));
% % % % % % 
% % % % % % Pr(keep_t_ts==2) = real(To(keep_t_ts==2)+(xs(keep_t_ts==2) - To(keep_t_ts==2)).*cos(pi./(w2(keep_t_ts==2)*4)) - sqrt(Ta(keep_t_ts==2).^2 - (xs(keep_t_ts==2)-To(keep_t_ts==2)).^2) .*sin(pi./(w2(keep_t_ts==2)*4)));
% % % % % % 
% % % % % % Pr(keep_t_ts==1) = real(To(keep_t_ts==1) + ((xs(keep_t_ts==1)-To(keep_t_ts==1)).*cos(pi./w2(keep_t_ts==1).*(ts(keep_t_ts==1)-t(keep_t_ts==1))) -sqrt(Ta(keep_t_ts==1).^2 - (xs(keep_t_ts==1)-To(keep_t_ts==1)).^2).*sin(pi./w2(keep_t_ts==1).*(ts(keep_t_ts==1)-t(keep_t_ts==1)))) .*exp(-(t(keep_t_ts==1) + 15/60 -ts(keep_t_ts==1))./k(keep_t_ts==1)));
% % % % % % %     else
% % % % % % %Pr(find(keep_t_ts==0)) = (To(find(keep_t_ts==0))+dT(find(keep_t_ts==0)))+(xs(find(keep_t_ts==0))-To(find(keep_t_ts==0))-dT(find(keep_t_ts==0))).*exp(-1/4./k(find(keep_t_ts==0)));
% % % % % % Pr(keep_t_ts==0) = To(keep_t_ts==0)+(xs(keep_t_ts==0)-To(keep_t_ts==0)).*exp(-1/4./k(keep_t_ts==0));
%     end


%x_(1:size(t,1),1:size(t,2),timestamp+1) = Pr;

% if sum(isnan(xs(:)))>0
%     disp('xs has Nan')
% end



% if ((tm < daySolarnoon(iday)) || (tm>daySolarnoon(iday)+6)) || (tm > ts - 1) || (Ta <=0) || ((ts <daySunset(iday)-3) || (ts > daySunset(iday)+3)) || (kTemp>100) || (w <0) %Use also reduced chi-square to select cycles
%         current_cycle_select = 0;
%         %Display them to check why a given cycle was removed from training model
%         %%%%%%%%%%%%%%%%%%%%%%%%%%
%         %tm                      %
%         %ts                      %
%         %daySolarnoon(iday)      %
%         %daySunset(iday)         %
%         %k                       %
%         %%%%%%%%%%%%%%%%%%%%%%%%%%
%     else
%         current_cycle_select = 1;
% end


% MULTIPLIER = ones(6,1,450);
% MULTIPLIER(2:6,1,:) = MULTIPLIER(2:6,1,:)/8;
%MULTIPLIER(4,1,:) = MULTIPLIER(4,1,:)/8;

% x_ = xs + process_noise.*MULTIPLIER; %/8;
%x_ = xs + process_noise./extend(3*sqrt(MULTIPLIER),Ne); %/17.5524; %/9.5530; %*0; %/10.3459; %/18.3707; %/8;
%x_ = xs + process_noise;
x_ = xs + process_noise./extend(sqrt(2)*erfinv(1-1/numel(xs))*sqrt(MULTIPLIER),Ne); 

% size(x_,2)
% size(tNoon)
% size(tSet)

% % % % % % % % % % % % % % % for i = 1:size(x_,2) %Number of pixels
% % % % % % % % % % % % % % %     %KeepJRemoved = [];
% % % % % % % % % % % % % % %     x_temp = squeeze(x_(:,i,:));
% % % % % % % % % % % % % % %     %size(x_temp)
% % % % % % % % % % % % % % %     member_removed = 0;
% % % % % % % % % % % % % % %     j = 1;
% % % % % % % % % % % % % % %     while j<= Ne - member_removed
% % % % % % % % % % % % % % % %         x_temp(3,j) 
% % % % % % % % % % % % % % % %         tNoon(i) 
% % % % % % % % % % % % % % % %         x_temp(4,j)
% % % % % % % % % % % % % % % %         x_temp(2,j)  
% % % % % % % % % % % % % % % %         tSet(i)
% % % % % % % % % % % % % % % %         x_temp(6,j)
% % % % % % % % % % % % % % % %         x_temp(5,j) %Use also reduced chi-square to select cycles
% % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % %        if  (x_temp(3,j) > x_temp(4,j) - 1) %|| ((x_temp(4,j) <tSet(i)-3) || (x_temp(4,j) > tSet(i)+3)) || ((x_temp(6,j)/pi.*(1./tan(pi./x_temp(6,j).*(x_temp(4,j)-x_temp(3,j)))))>100) || (x_temp(6,j) <0 || x_temp(6,j) > 12) || ((x_temp(3,j) < tNoon(i)) || (x_temp(3,j)>tNoon(i)+6)) || (x_temp(5,j) <0 || x_temp(5,j) > 15) %|| (x_temp(5,j) + x_temp(6,j))>24
% % % % % % % % % % % % % % %   
% % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % %        %if ((x_temp(3,j) < tNoon(i)) || (x_temp(3,j)>tNoon(i)+6)) || (x_temp(3,j) > x_temp(4,j) - 1) || (x_temp(2,j) <=0) || ((x_temp(4,j) <tSet(i)-3) || (x_temp(4,j) > tSet(i)+3)) || ((x_temp(6,j)/pi.*(1./tan(pi./x_temp(6,j).*(x_temp(4,j)-x_temp(3,j)))))>100) || (x_temp(5,j) <0) %|| (x_temp(6,j) <0 || x_temp(6,j) > 15)%Use also reduced chi-square to select cycles
% % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % %         %if ((x_temp(3,j) < tNoon(i)) || (x_temp(3,j)>tNoon(i)+6)) || (x_temp(3,j) > x_temp(4,j) - 1) || (x_temp(2,j) <=0) || ((x_temp(4,j) <tSet(i)-3) || (x_temp(4,j) > tSet(i)+3)) || ((x_temp(6,j)/pi.*(1./tan(pi./x_temp(6,j).*(x_temp(4,j)-x_temp(3,j)))))>100) || (x_temp(5,j) <0 || x_temp(5,j) > 15) || (x_temp(6,j) <0 || x_temp(6,j) > 15)%Use also reduced chi-square to select cycles
% % % % % % % % % % % % % % %         %if ((x_(3,i,j) < tNoon(i)) || (x_(3,i,j)>tNoon(i)+6)) || (x_(3,i,j) > x_(4,i,j) - 1) || (x_(2,i,j) <=0) || ((x_(4,i,j) <tSet(i)-3) || (x_(4,i,j) > tSet(i)+3)) || ((x_(6,i,j)/pi.*(1./tan(pi./x_(6,i,j).*(x_(4,i,j)-x_(3,i,j)))))>100) || (x_(5,i,j) <0) %Use also reduced chi-square to select cycles
% % % % % % % % % % % % % % %             %if ((tm < daySolarnoon(iday)) || (tm>daySolarnoon(iday)+6)) || (tm > ts - 1) || (Ta <=0) || ((ts <daySunset(iday)-3) || (ts > daySunset(iday)+3)) || (kTemp>100) || (w <0) %Use also reduced chi-square to select cycles
% % % % % % % % % % % % % % %             %KeepJRemoved = [KeepJRemoved j]; 
% % % % % % % % % % % % % % %             x_temp(:,j) = [];
% % % % % % % % % % % % % % %             member_removed = member_removed + 1;
% % % % % % % % % % % % % % %             j = j - 1;
% % % % % % % % % % % % % % %         end
% % % % % % % % % % % % % % %         j = j + 1;
% % % % % % % % % % % % % % %     end
% % % % % % % % % % % % % % %     mx_temp = mean(x_temp,2);
% % % % % % % % % % % % % % %     x_temp(:,size(x_temp,2)+1:size(x_,3)) = mx_temp*ones(1,size(x_,3)-size(x_temp,2)); %Feel with the mean
% % % % % % % % % % % % % % %     %size(x_temp)
% % % % % % % % % % % % % % %     x_(1:size(x_,1),i,1:size(x_,3)) = x_temp;
% % % % % % % % % % % % % % % end
%      To = x_(1,i,j)
%      Ta = x_(2,i,j)
%      tm = x_(3,i,j)
%      ts = x_(4,i,j)
%      w1 = x_(5,i,j)
%      w2 = x_(6,i,j)
     
%k = w2/pi.*(1./tan(pi./w2.*(ts-tm)));%-dT/Ta*(1/sin(pi/w*(ts-tm))));  %pi/w*(ts-tm) must be in [-1,1]
%k = x_(6,i,j)/pi.*(1./tan(pi./x_(6,i,j).*(x_(4,i,j)-x_(3,i,j))));%-dT/Ta*(1/sin(pi/w*(ts-tm))));  %pi/w*(ts-tm) must be in [-1,1]





% if sum(isnan(x_(:)))>0
%     disp('x_ has Nan')
% end


% if sum(isnan(xs))>0
%     disp('xs has Nan')
% end




% if sum(isnan(process_noise))>0
%  disp('process noise Nan')
% end
%x_(1:size(t,1),1:size(t,2),timestamp) = Pr; %for sequential_v
%end

%I(:,:) = eye(M,M);
%Kn = (I - Gf * C).*Kn_(:,:,timestamp); %Correlation matrix of error in the filtered estimate
EnsPXs = xs - extend(mean(xs,3),Ne);
%Kn_(1:size(t,1),1:size(t,2),timestamp) = calculateForecastErrorCovariance(EnsPX_);
Kn = calculateForecastErrorCovariance(EnsPXs);



%Kn_(1:size(t,1),1:size(t,2),timestamp+1) = F.* Kn .*F + Q1; %Correlation matrix of error in the predicted estimate for the next sample
EnsPX_ = x_ - extend(mean(x_,3),Ne);
%Kn_(1:size(t,1),1:size(t,2),timestamp) = calculateForecastErrorCovariance(EnsPX_);
Kn_ = calculateForecastErrorCovariance(EnsPX_);

prediction(:,:,timestamp+1) = mean(x_,3);

% if sum(isnan(x_(:)))>0
%     disp('x_ has Nan')
% end
% if sum(isnan(prediction(:,:,timestamp+1)))>0
%     disp('prediction has Nan')
% end
predictedObservation(:,:,timestamp+1) = observationFunction(prediction(:,:,timestamp+1),y,t+15/60);
%predictedc(:,:,timestamp+1) = thermalsunriseFunction(prediction(:,:,timestamp+1),c,t+15/60,tHorizon);


% if sum(isnan(predictedObservation(:)))>0
%     disp('predictedObservation has Nan')
% end


%     KKK = Kn(n)
%     KKK_ = Kn_(n)
%     SSS = S(n)

% S(S==0) = eps; %For S= 0, I replaced it with a small value 2 x (2.Pixels)
% size(inn)
% size(S)
stResidual = inn./sqrt(reshape(S(1,1:2:end),size(inn,1),size(inn,2))); %Standardized residual

