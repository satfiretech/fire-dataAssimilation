function [KeepFireDetect x_i Kn_ KeepAlpha Keep_S stResidual ye cycle_out_scope cycle_reported xs Gf Keep_Q2a EnsPX_ prediction predictedObservation Bo KeepJJ te Keepxs he Rea FEATURE1 y_predictedKeep] = stdw4DVarDECONV10_v3(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_i,Kn_,KeepAlpha,Keep_S,KeepFireDetect,DECONV,restored,Keep_Q2a,ASSIMILATION_WINDOW_LENGTH, EnsPX_, prediction, predictedObservation, Bo, KeepJJ, Keepxs, threshold, FEATURE1, RESIDUAL, NUMBER_ENSEMBLE_MEMBERS, y_predictedKeep)
%function [KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual y cycle_out_scope cycle_reported xs Gf Keep_Q2] = stdw4DVarDECONV10_v3(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,DECONV,restored,Keep_Q2,ASSIMILATION_WINDOW_LENGTH)
%function [KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual y cycle_out_scope cycle_reported xs Gf Keep_Q2 EnsPX_ prediction predictedObservation] = stdxtkalmanDECONV10(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,DECONV,restored,Keep_Q2,NUMBER_ENSEMBLE_MEMBERS, EnsPX_, prediction, predictedObservation)
%KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual y cycle_out_scope cycle_reported xs Gf Keep_Q2 EnsPX_ prediction predictedObservation


stResidual = [];
Gf = [];

%TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,DECONV,restored,Keep_Q2,ASSIMILATION_WINDOW_LENGTH,EnsPX_, prediction, predictedObservation
%TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,DECONV,restored,Keep_Q2,NUMBER_ENSEMBLE_MEMBERS, EnsPX_, prediction, predictedObservation

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


v4 = zeros(NUMBER_ROW,NUMBER_COLUMN); %Increase the dimension
t = zeros(NUMBER_ROW,NUMBER_COLUMN); %Increase the dimension
w1 = zeros(NUMBER_ROW,NUMBER_COLUMN);
w2 = zeros(NUMBER_ROW,NUMBER_COLUMN);
To = zeros(NUMBER_ROW,NUMBER_COLUMN);
Ta = zeros(NUMBER_ROW,NUMBER_COLUMN);
tm = zeros(NUMBER_ROW,NUMBER_COLUMN);
ts = zeros(NUMBER_ROW,NUMBER_COLUMN);
%dT = zeros(NUMBER_ROW,NUMBER_COLUMN);
k = zeros(NUMBER_ROW,NUMBER_COLUMN);
%Q = zeros(NUMBER_ROW,NUMBER_COLUMN); %Increase the dimension
R1 = zeros(NUMBER_ROW,NUMBER_COLUMN); %Increase the dimension
cycle_out_scope = zeros(NUMBER_ROW,NUMBER_COLUMN);


for jN = 1:NUMBER_ROW
    for iN = 1:NUMBER_COLUMN
        %         for wN = ASSIMILATION_WINDOW_LENGTH:-1:1
        %             nW = n - wN;
        %             v4(jN,iN,wN) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_DATA(nW(jN,iN));
        %         end
        %         v4(jN,iN,ASSIMILATION_WINDOW_LENGTH+1) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_DATA(n(jN,iN));
        wN = 1;
        inF = 0;
        while inF < ASSIMILATION_WINDOW_LENGTH
            nW = n - wN;
            if KeepFireDetect(jN,iN,timestamp-wN)==0
                v4(jN,iN,ASSIMILATION_WINDOW_LENGTH-inF) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_DATA(nW(jN,iN));
                inF = inF + 1;
            end
            wN = wN + 1;
        end
        v4(jN,iN,ASSIMILATION_WINDOW_LENGTH+1) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_DATA(n(jN,iN));

        %         for wN = ASSIMILATION_WINDOW_LENGTH:-1:1
        %             nW = n - wN;
        %             t(jN,iN,wN) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME(nW(jN,iN));
        %         end
        %         t(jN,iN,ASSIMILATION_WINDOW_LENGTH+1) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME(n(jN,iN));
        %         nW = n + 1;
        %         t(jN,iN,ASSIMILATION_WINDOW_LENGTH+2) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME(nW(jN,iN));
        wN = 1;
        inF = 0;
        while inF < ASSIMILATION_WINDOW_LENGTH
            nW = n - wN;
            if KeepFireDetect(jN,iN,timestamp-wN)==0
                t(jN,iN,ASSIMILATION_WINDOW_LENGTH-inF) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME(nW(jN,iN));
                inF = inF + 1;
            end
            wN = wN + 1;
        end
        t(jN,iN,ASSIMILATION_WINDOW_LENGTH+1) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME(n(jN,iN));
        nW = n + 1;
        t(jN,iN,ASSIMILATION_WINDOW_LENGTH+2) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME(nW(jN,iN));


        w1(jN,iN) = TESTINGCYCLE_DATA(jN,iN).w(n(jN,iN));
        w2(jN,iN) = TESTINGCYCLE_DATA(jN,iN).w2(n(jN,iN));
        To(jN,iN) = TESTINGCYCLE_DATA(jN,iN).To(n(jN,iN));
        Ta(jN,iN) = TESTINGCYCLE_DATA(jN,iN).Ta(n(jN,iN));
        tm(jN,iN) = TESTINGCYCLE_DATA(jN,iN).tm(n(jN,iN));
        ts(jN,iN) = TESTINGCYCLE_DATA(jN,iN).ts(n(jN,iN));
        %dT(jN,iN) = TESTINGCYCLE_DATA(jN,iN).dT(n(jN,iN));
        k(jN,iN) = TESTINGCYCLE_DATA(jN,iN).k(n(jN,iN));

        %Q(jN,iN) = TESTINGCYCLE_DATA(jN,iN).Q(n(jN,iN));

        %         for wN = ASSIMILATION_WINDOW_LENGTH+1:-1:1
        %             nW = n - wN;
        %             R1(jN,iN,wN) = TESTINGCYCLE_DATA(jN,iN).R1(nW(jN,iN));
        %         end
        %         R1(jN,iN,ASSIMILATION_WINDOW_LENGTH+1) = TESTINGCYCLE_DATA(jN,iN).R1(n(jN,iN));
        wN = 1;
        inF = 0;
        while inF < ASSIMILATION_WINDOW_LENGTH
            nW = n - wN;
            if KeepFireDetect(jN,iN,timestamp-wN)==0
                R1(jN,iN,ASSIMILATION_WINDOW_LENGTH-inF) = TESTINGCYCLE_DATA(jN,iN).R1(nW(jN,iN));
                R1c(jN,iN,ASSIMILATION_WINDOW_LENGTH-inF) = TESTINGCYCLE_DATA(jN,iN).R1c(nW(jN,iN));
                inF = inF + 1;
            end
            wN = wN + 1;
        end
        R1(jN,iN,ASSIMILATION_WINDOW_LENGTH+1) = TESTINGCYCLE_DATA(jN,iN).R1(n(jN,iN));
        R1c(jN,iN,ASSIMILATION_WINDOW_LENGTH+1) = TESTINGCYCLE_DATA(jN,iN).R1c(n(jN,iN));




        %cycle_out_scope(jN,iN) = TESTINGCYCLE_DATA(jN,iN).TREATED(n(jN,iN));
        %cycle_out_scope(jN,iN) = []; %TESTINGCYCLE_DATA(jN,iN).TREATED(n(jN,iN));


        %         for wN = ASSIMILATION_WINDOW_LENGTH:-1:1
        %             nW = n - wN;
        %             tHorizon(jN,iN,wN) = TESTINGCYCLE_DATA(jN,iN).tmorningHorizon(nW(jN,iN));
        %             tThermalSunrise(jN,iN,wN) = TESTINGCYCLE_DATA(jN,iN).tSunriseDay(nW(jN,iN));
        %         end
        %         tHorizon(jN,iN,ASSIMILATION_WINDOW_LENGTH+1) = TESTINGCYCLE_DATA(jN,iN).tmorningHorizon(n(jN,iN));
        %         tThermalSunrise(jN,iN,ASSIMILATION_WINDOW_LENGTH+1) = TESTINGCYCLE_DATA(jN,iN).tSunriseDay(n(jN,iN));
        wN = 1;
        inF = 0;
        while inF < ASSIMILATION_WINDOW_LENGTH
            nW = n - wN;
            if KeepFireDetect(jN,iN,timestamp-wN)==0
                tHorizon(jN,iN,ASSIMILATION_WINDOW_LENGTH-inF) = TESTINGCYCLE_DATA(jN,iN).tmorningHorizon(nW(jN,iN));
                tThermalSunrise(jN,iN,ASSIMILATION_WINDOW_LENGTH-inF) = TESTINGCYCLE_DATA(jN,iN).tSunriseDay(nW(jN,iN));
                inF = inF + 1;
            end
            wN = wN + 1;
        end
        tHorizon(jN,iN,ASSIMILATION_WINDOW_LENGTH+1) = TESTINGCYCLE_DATA(jN,iN).tmorningHorizon(n(jN,iN));
        tThermalSunrise(jN,iN,ASSIMILATION_WINDOW_LENGTH+1) = TESTINGCYCLE_DATA(jN,iN).tSunriseDay(n(jN,iN));

    end
end
c = tThermalSunrise - tHorizon; %Row x Column x Day
%R1c = R1;

%CHANGE HERE for Q, it must be trained.
for iN = 1:NUMBER_COLUMN
    for jN = 1:NUMBER_ROW
        %Q(1:7,1+((jN + NUMBER_ROW*(iN-1))-1)*7:7+((jN + NUMBER_ROW*(iN-1))-1)*7)
        %Q(1:6,1+((jN + NUMBER_ROW*(iN-1))-1)*6:6+((jN + NUMBER_ROW*(iN-1))-1)*6) = TESTINGCYCLE_DATA(jN,iN).Q(n(jN,iN))/10*eye(6); %1
        %Q(:,(iN-1)+1:(iN-1)+7)
        Q(1:6,1+((jN + NUMBER_ROW*(iN-1))-1)*6:6+((jN + NUMBER_ROW*(iN-1))-1)*6) = TESTINGCYCLE_DATA(jN,iN).CovParameter;
    end
end %[(7 . 7) x Pixels]  6 x (6.Pixels)
%Q is assumed constant for a given pixel
%Q = [zeros(7) zeros(7) zeros(7) zeros(7) zeros(7) zeros(7)];
%Qb = (blockdiag(Q,N+1)).'; [block diagonal]






%Ne = NUMBER_ENSEMBLE_MEMBERS;
%%%
%%%

if DECONV==1
    y = restored; %Consider the window
else
    y = v4; %I can optimize the code by not reading v4 in case of DECONV = 1

    for nW = 1:ASSIMILATION_WINDOW_LENGTH+1
        yblock(nW,:) = reshape(y(:,:,nW),1,size(y,1)*size(y,2));  %[Day x Pixels]

        %     yblock = [reshape(y(:,:,1),1,size(y,1)*size(y,2));
        %         reshape(y(:,:,2),1,size(y,1)*size(y,2));
        %         reshape(y(:,:,3),1,size(y,1)*size(y,2));
        %         reshape(y(:,:,4),1,size(y,1)*size(y,2));
        %         reshape(y(:,:,5),1,size(y,1)*size(y,2));
        %         reshape(y(:,:,6),1,size(y,1)*size(y,2));
        %reshape(y(:,:,1),1,size(y,1)*size(y,2))];

        tblock(nW,:) = reshape(t(:,:,nW),1,size(t,1)*size(t,2));

        cblock(nW,:) = reshape(c(:,:,nW),1,size(c,1)*size(c,2));
        tHorizonblock(nW,:) = reshape(tHorizon(:,:,nW),1,size(tHorizon,1)*size(tHorizon,2));
    end
end

Q1 = Q; %Process noise covariance
% y(:,:,timestamp) = v4;
% Q1(:,:,timestamp) = Q;

yoblock = [];
for nW = 1:ASSIMILATION_WINDOW_LENGTH+1
    yoblock = [yoblock;yblock(nW,:);cblock(nW,:)]; %(2.Day) x Pixels
end
%%%
%%%

xs = zeros(6,NUMBER_ROW*NUMBER_COLUMN); %xf is filtered samples, it will keep all filtered distribution, each distribution on a each pixels
process_noise = normalizedState(Q1,randn(size(xs))); %[Number of variables x Number of pixels] 6 x Pixels


%Check why is w1 not w2 also
if timestamp==ASSIMILATION_WINDOW_LENGTH+1


    %CHANGE HERE for Bo = Pb, it must be trained. Background error covariance matrix.
    %It is constant for a given pixel.
    %Bo = Q*10; %Ao + Q
    %Bo = Q/10; %*10; %Ao + Q
    BBo = [9.8178   -5.5937   -0.0562    0.2861   -2.4879    0.2050
        -5.5937    9.6852    0.1932   -0.7335    1.3157    0.7264
        -0.0562    0.1932    0.0938   -0.0529    0.2396   -0.2255
        0.2861   -0.7335   -0.0529    0.4740   -0.2062    0.3343
        -2.4879    1.3157    0.2396   -0.2062    1.8044   -0.8073
        0.2050    0.7264   -0.2255    0.3343   -0.8073    1.9948];
    BBo = diag(diag(BBo));

    %Bo = [BBo BBo BBo BBo BBo BBo BBo BBo BBo];
    Bo = repmat(BBo,1,NUMBER_ROW*NUMBER_COLUMN); %6 x (6.Pixels)
    %Pbo = rand(7,7);
    %Bo = Pbo;
    for wN = 1:ASSIMILATION_WINDOW_LENGTH
        nW = n - wN;
% % % % % % % % % % % % % % % %         for jN = 1:NUMBER_ROW
% % % % % % % % % % % % % % % %             for iN = 1:NUMBER_COLUMN
% % % % % % % % % % % % % % % %                 w11(jN,iN) = TESTINGCYCLE_DATA(jN,iN).w(nW(jN,iN));
% % % % % % % % % % % % % % % %                 w21(jN,iN) = TESTINGCYCLE_DATA(jN,iN).w2(nW(jN,iN));
% % % % % % % % % % % % % % % %                 To1(jN,iN) = TESTINGCYCLE_DATA(jN,iN).To(nW(jN,iN));
% % % % % % % % % % % % % % % %                 Ta1(jN,iN) = TESTINGCYCLE_DATA(jN,iN).Ta(nW(jN,iN));
% % % % % % % % % % % % % % % %                 tm1(jN,iN) = TESTINGCYCLE_DATA(jN,iN).tm(nW(jN,iN));
% % % % % % % % % % % % % % % %                 ts1(jN,iN) = TESTINGCYCLE_DATA(jN,iN).ts(nW(jN,iN));
% % % % % % % % % % % % % % % %                 %dT(jN,iN) = TESTINGCYCLE_DATA(jN,iN).dT(n(jN,iN));
% % % % % % % % % % % % % % % %                 %k(jN,iN) = TESTINGCYCLE_DATA(jN,iN).k(n(jN,iN));
% % % % % % % % % % % % % % % %             end
% % % % % % % % % % % % % % % %         end
% % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % %         x_i = [reshape(To1,1,size(To,1)*size(To,2));  %[6 x pixels] %Start of the assimilation window
% % % % % % % % % % % % % % % %             reshape(Ta1,1,size(Ta,1)*size(Ta,2));
% % % % % % % % % % % % % % % %             reshape(tm1,1,size(tm,1)*size(tm,2));
% % % % % % % % % % % % % % % %             reshape(ts1,1,size(ts,1)*size(ts,2));
% % % % % % % % % % % % % % % %             reshape(w11,1,size(w1,1)*size(w1,2));
% % % % % % % % % % % % % % % %             reshape(w21,1,size(w2,1)*size(w2,2))];
        %reshape(c,1,size(c,1)*size(c,2))];

        
            x_i = [reshape(To,1,size(To,1)*size(To,2));  %[6 x pixels]
        reshape(Ta,1,size(Ta,1)*size(Ta,2));
        reshape(tm,1,size(tm,1)*size(tm,2));
        reshape(ts,1,size(ts,1)*size(ts,2));
        reshape(w1,1,size(w1,1)*size(w1,2));
        reshape(w2,1,size(w2,1)*size(w2,2))];
    %reshape(c,1,size(c,1)*size(c,2))];

        
        %y_extended = extend(y(:,:,nW),NUMBER_ENSEMBLE_MEMBERS);
        y_predicted = observationFunction(x_i,y(:,:,ASSIMILATION_WINDOW_LENGTH - wN +1),t(:,:,ASSIMILATION_WINDOW_LENGTH - wN +1)); %CORRECT y_extended   Nbre Rows x Nbre Columns
        %y_predicted = observationFunction(x_i,y(:,:,nW),t(:,:,nW)); %CORRECT y_extended
        forecastSteps = 1;
        y_predictedkeep(forecastSteps,1:size(y_predicted,1)*size(y_predicted,2),1:size(y_predicted,3)) = reshape(y_predicted,1,size(y_predicted,1)*size(y_predicted,2),size(y_predicted,3));
        ypredictedMean = mean(y_predictedkeep,3); %Ensemble
        y_predictedKeep(:,:,ASSIMILATION_WINDOW_LENGTH - wN +1) = reshape(ypredictedMean(1,:),size(y,1),size(y,2));

    end




% % % % % % % % % % % % % % % %     xi = [reshape(To,1,size(To,1)*size(To,2));  %[6 x pixels]
% % % % % % % % % % % % % % % %         reshape(Ta,1,size(Ta,1)*size(Ta,2));
% % % % % % % % % % % % % % % %         reshape(tm,1,size(tm,1)*size(tm,2));
% % % % % % % % % % % % % % % %         reshape(ts,1,size(ts,1)*size(ts,2));
% % % % % % % % % % % % % % % %         reshape(w1,1,size(w1,1)*size(w1,2));
% % % % % % % % % % % % % % % %         reshape(w2,1,size(w2,1)*size(w2,2))];
% % % % % % % % % % % % % % % %     %reshape(c,1,size(c,1)*size(c,2))];


    %x_(1:size(t,1),1:size(t,2),timestamp) = To+Ta.*cos(pi./w1.*(t-tm)); %xs; %TO CHECK
    %x_= extend(xi,Ne) + process_noise;
% % % % % % % % % % % % % % % %     x_f= xi; %+ process_noise; %xb= background state at launch time of the 4DVar [No noise is added]
    x_f= x_i; %+ process_noise; %xb= background state at launch time of the 4DVar [No noise is added]
        %End of the assimilation window
    %%%%%%
    %%%%%%
    %%%%%%

    %     y_predicted = observationFunction(x_f,y(:,:,ASSIMILATION_WINDOW_LENGTH +1),t(:,:,ASSIMILATION_WINDOW_LENGTH +1)); %CORRECT y_extended
    %     %y_predicted = observationFunction(x_i,y(:,:,nW),t(:,:,nW)); %CORRECT y_extended
    %     forecastSteps = 1;
    %     y_predictedkeep(forecastSteps,1:size(y_predicted,1)*size(y_predicted,2),1:size(y_predicted,3)) = reshape(y_predicted,1,size(y_predicted,1)*size(y_predicted,2),size(y_predicted,3));
    %     ypredictedMean = mean(y_predictedkeep,3); %Ensemble
    %     %y_predictedKeep(:,:,ASSIMILATION_WINDOW_LENGTH +1) = reshape(ypredictedMean(1,:),size(y,1),size(y,2));
    %     YPPP = reshape(ypredictedMean(1,:),size(y,1),size(y,2))


    %%%%%%
    %%%%%%
    %%%%%%




    %     length(find(isnan(x_)==1))

    %     %Kn_(1:size(t,1),1:size(t,2),timestamp) = Q; %10;%xcorr
    %     EnsPX_ = x_ - extend(mean(x_,3),Ne);
    %     %Kn_(1:size(t,1),1:size(t,2),timestamp) = calculateForecastErrorCovariance(EnsPX_);
    %     Kn_ = calculateForecastErrorCovariance(EnsPX_);
    %     prediction(:,:,timestamp) = mean(x_,3);
    prediction(:,:,timestamp) = x_f;
    %prediction(:,:,1)
    %mean(process_noise,3)

    predictedObservation(:,:,timestamp) = observationFunction(prediction(:,:,timestamp),y(:,:,ASSIMILATION_WINDOW_LENGTH+1),t(:,:,ASSIMILATION_WINDOW_LENGTH+1)); %Row x Column x timestamp
    % else
    %     Q = Bo/2;
    %     Q1 = Q;
    %    Bo = Q*2;
end




%C = 1;%deltaCnl/deltax at x = xn_
%F;
%F = [nan(7) eye(7) eye(7) eye(7) eye(7) eye(7)];
%Fb = (blockdiag(F,N+1)).'; [block diagonal]
F = 1;


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

%y_extended = extend(y,Ne);
%y_perturbed = y_extended + extend(sqrt(Q2),Ne).*randn(size(y_extended));
%disp('1')
%y_predicted = observationFunction(x_,y_perturbed,t);
%y_predicted = observationFunction(x_,y_perturbed,t);

% predictedObservation(:,:,timestamp) = observationFunction(prediction(:,:,timestamp),y(:,:,1),t(:,:,1));
%[row x colum x timestamp]
inn = y(:,:,ASSIMILATION_WINDOW_LENGTH+1)-predictedObservation(:,:,timestamp); %then, this gives 50 x 50 [row x column]

% alpha = y_perturbed - y_predicted; %1 x 2500 x Ne for a 50 x 50 images
% inn = y-reshape(mean(y_predicted,3),size(y,1),size(y,2)); %then, this gives 50 x 50


% if sum(isnan(x_))>0
%     disp('x_ has Nan')
% end


% if sum(isnan(y_predicted))>0
%     disp('y_predicted has Nan')
% end




%KeepAlpha(1:size(t,1),1:size(t,2),timestamp) = alpha;


% KeepAlpha(1:size(t,1),1:size(t,2),timestamp) = alpha;

% disp('2')
% HX = y_predicted - extend(mean(y_predicted,3),Ne);
% HPH = sum(HX.*HX,3)/(Ne-1);
%S = C.*Kn_(1:size(t,1),1:size(t,2),timestamp).*C'+Q2; %Innovation or Residual covariance
%Gf = Kn_(:,:,timestamp).*C'.*(S).^(-1); %Kalman gain %THIS CAN BE REMOVED AS IS CALCULATED AFTER DETECTION
% S = reshape(HPH,size(Q2)) + Q2;


% Keep_S(1:size(t,1),1:size(t,2),timestamp) = S;
Keep_Q2(1:size(t,1),1:size(t,2),timestamp) = Q2(:,:,ASSIMILATION_WINDOW_LENGTH+1); %Keep the measurement covariance

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


if  length(find(isnan(Q2(:,:,ASSIMILATION_WINDOW_LENGTH+1))==1))~=0
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

%     SRQ2 = restoreMissingSamples10(sqrt(Q2(:,:,ASSIMILATION_WINDOW_LENGTH+1)),timestamp,windowSize_row,windowSize_column,window_start_row, window_start_column,sqrt(Keep_Q2));
%     SRQ2(SRQ2<0) = 0;
%     Q2(:,:,ASSIMILATION_WINDOW_LENGTH+1) = SRQ2.^2;
    %S = restoreMissingSamples6(S,timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,Keep_S);
    %S = Kn_(:,:,timestamp) + Q2;
    %Kn_(:,:,timestamp) = S - Q2; %Considering that C is a scalar equal to 1
end
%S = Kn_(:,:,timestamp) + Q2;
%Kn_(:,:,timestamp) = S  - Q2;
% S = HPH + Q2;


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
Keep_Q2(1:size(t,1),1:size(t,2),timestamp) = Q2(:,:,ASSIMILATION_WINDOW_LENGTH+1); %Update measurement covariance
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
% S = reshape(HPH,size(Q2)) + Q2;
% S(S==0) = eps;
%Gf = mod(fire_detect + 1,2).*Kn_(:,:,timestamp).*C'.*(S).^(-1); %Done two times, abuse CASE OF NO FIRE (Fire=1, no fire = 0)
% PH = backgroundValueOfObservation(EnsPX_,HX);
%Gf = mod(fire_detect + 1,2).*PH.*(reshape(S,1,size(S,1)*size(S,2))).^(-1); %Done two times, abuse CASE OF NO FIRE (Fire=1, no fire = 0)


% Gf = PH.*(reshape(S,1,size(S,1)*size(S,2))).^(-1); %Done two times, abuse CASE OF NO FIRE (Fire=1, no fire = 0)

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




%NAVDAS-AR
%First iteration
j=1;
%--------
%--------

%Outer loop
%-----------
%deltaxp(:,n,0) = xp(:,n,0) - x(:,n,-1) or deltaxp(:,n,0) = 0
%deltaxp = 0

%Inner loop
%----------
%step1 (outer loop)
%xb = x_
xsf = zeros(size(x_i,1),size(x_i,2),ASSIMILATION_WINDOW_LENGTH+1);
%size(x_)
%size(xsf)
xsf(:,:,1) = x_i; %x^b_0
for i = 2:ASSIMILATION_WINDOW_LENGTH+1
    %xsf = [xsf modelForecast(xsf(:,end))];
    xsf(:,:,i) = modelForecast(xsf(:,:,i-1)); %[6 x Pixels x Days]: 6 = ASSIMILATION_WINDOW_LENGTH+1
end

for i = 1:size(x_i,2)
    xsfvector(:,i) = reshape(xsf(:,i,:),size(xsf,1)*size(xsf,3),1); %[(6.Days) x Pixels]
end
%xsfvector = (reshape(xsf,1,size(xsf,1)*size(xsf,2))).'; %[42 x Pixels]




% Q2x = reshape(R1,1,size(R1,1)*size(R1,2));
% Q2c = reshape(R1c,1,size(R1c,1)*size(R1c,2));
% for i = 1:size(R1,1)*size(R1,2)
%     Q2a(1,1+(i-1)*2:2+(i-1)*2) = [Q2x(i) 0];
%     Q2a(2,1+(i-1)*2:2+(i-1)*2) = [0 Q2c(i)];
% end


%Q2Day = R
for nW = 1:ASSIMILATION_WINDOW_LENGTH+1  %Consideration of uncorrelation between y and c, the memory usage can be reduced as done next
    %     Q2Day(nW,:) = reshape(Q2(:,:,nW),1,size(Q2,1)*size(Q2,2));  %[Day x Pixel]

    Q2Day((nW-1)*2+1:nW*2,:) = [reshape(Q2(:,:,nW),1,size(Q2,1)*size(Q2,2)); reshape(R1c(:,:,nW),1,size(R1c,1)*size(R1c,2))]; %[(Day.2) x Pixels]
    %     Q2cDay(nW,:) = reshape(R1c(:,:,nW),1,size(R1c,1)*size(R1c,2));  %[Day x Pixel]

    %     yblock = [reshape(y(:,:,1),1,size(y,1)*size(y,2));
    %         reshape(y(:,:,2),1,size(y,1)*size(y,2));
    %         reshape(y(:,:,3),1,size(y,1)*size(y,2));
    %         reshape(y(:,:,4),1,size(y,1)*size(y,2));
    %         reshape(y(:,:,5),1,size(y,1)*size(y,2));
    %         reshape(y(:,:,6),1,size(y,1)*size(y,2));
    %reshape(y(:,:,1),1,size(y,1)*size(y,2))];
end


for i = 1:size(Q2Day,2)
    Q2a(1,1+(i-1)*2:2+(i-1)*2) = [Q2Day(2*ASSIMILATION_WINDOW_LENGTH+1,i) 0];
    Q2a(2,1+(i-1)*2:2+(i-1)*2) = [0 Q2Day(2*(ASSIMILATION_WINDOW_LENGTH + 1),i)];
end %[2x(2.Pixels)]
Keep_Q2a(:,:,timestamp) = Q2a;

%Get fire detection result at launch of the 4DVar
if timestamp== ASSIMILATION_WINDOW_LENGTH + 1
%     for pixelnumber = 1:size(x_i,2)  %THIS MUST BE CHANGED
%         %         hto = Ht(1:6,pixelnumber,1);
%         %         htc1 = Ht(7:12,pixelnumber,1);
%         %         hta = [hto.';htc1.'];
%         %         %R1a = Rta = Q2Day(1:2,:);
%         %         %CovObs = hta*Bo(1:6,1+(pixelnumber-1)*6:pixelnumber*6)*hta.' + diag(Rta(:,pixelnumber));
%         %         CovObs = hta*Bo(1:6,1+(pixelnumber-1)*6:pixelnumber*6)*hta.' + diag(Q2Day(1:2,pixelnumber));
%         %         Ao(1:6,1+(pixelnumber-1)*6:pixelnumber*6) = (eye(6,6) - Bo(1:6,1+(pixelnumber-1)*6:pixelnumber*6)*hta.'*inv(CovObs)*hta)*Bo(1:6,1+(pixelnumber-1)*6:pixelnumber*6);
%     end
    %     filtered1 = createPerturbedAnalysisMC(xs,NUMBER_ENSEMBLE_MEMBERS,Ao);
    prediction1 = createPerturbedAnalysisMC(x_f,NUMBER_ENSEMBLE_MEMBERS,Bo);

    %     display('Dimensions')
    %     size(filtered1)
    %     size(prediction1)
    %     size(Q2Day)
    %     size(t(:,:,1))
    %     size(y(:,:,1))

    if RESIDUAL == 0
        [FIREFLAG FEATURE1 y_predictedM] = startTimeFireDetect(y(:,:,ASSIMILATION_WINDOW_LENGTH+1), Keep_Q2a, prediction1, t(:,:,ASSIMILATION_WINDOW_LENGTH+1), threshold, RESIDUAL);
        KeepFireDetect(:,:,timestamp) = FIREFLAG;
        y_predictedKeep(:,:,timestamp) = y_predictedM;

    elseif RESIDUAL == 1
        [FIREFLAG FEATURE1 y_predictedM] = startTimeFireDetect(y(:,:,ASSIMILATION_WINDOW_LENGTH+1), Keep_Q2a, prediction1, t(:,:,ASSIMILATION_WINDOW_LENGTH+1), threshold, RESIDUAL);
        KeepFireDetect(:,:,timestamp) = FIREFLAG;
        y_predictedKeep(:,:,timestamp) = y_predictedM;

    elseif RESIDUAL == 2
        [FIREFLAG FEATURE1 y_predictedM] = startTimeFireDetect(y(:,:,ASSIMILATION_WINDOW_LENGTH+1), Keep_Q2a, prediction1, t(:,:,ASSIMILATION_WINDOW_LENGTH+1), threshold, RESIDUAL);
        KeepFireDetect(:,:,timestamp) = FIREFLAG;
        y_predictedKeep(:,:,timestamp) = y_predictedM;

    else
        display('specify the feature')
    end
end
%--------------
% if KeepFireDetect(:,:,timestamp)
%     Q2Day = 1/eps*ones(size(Q2Day));
% end
% for wN = 1:ASSIMILATION_WINDOW_LENGTH+1
%     tempQ2 = Q2(:,:,wN);
%     tempQ2(KeepFireDetect(:,:,timestamp)==1) = 1/eps;
%     Q2(:,:,wN) = tempQ2;
%
%     tempR1c = R1c(:,:,wN);
%     tempR1c(KeepFireDetect(:,:,timestamp)==1) = 1/eps;
%     R1c(:,:,wN) = tempR1c;
% end







%Q2block = Rb
%for iN = 1:NUMBER_COLUMN
%    for jN = 1:NUMBER_ROW
for i = 1:size(Q2Day,2)
    %Q2Block(1:ASSIMILATION_WINDOW_LENGTH+1,1+((jN + NUMBER_ROW*(iN-1))-1)*7:7+((jN + NUMBER_ROW*(iN-1))-1)*7) = TESTINGCYCLE_DATA(jN,iN).Q(n(jN,iN))/10*eye(7); %1
    %Q2Block(1:ASSIMILATION_WINDOW_LENGTH+1,1+(i-1)*(ASSIMILATION_WINDOW_LENGTH+1):i*(ASSIMILATION_WINDOW_LENGTH+1)) = blockdiag(Q2Day(:,i).',ASSIMILATION_WINDOW_LENGTH+1); %1
    %Consideration of uncorrelation between y and c
    Q2Block(1:(ASSIMILATION_WINDOW_LENGTH+1)*2,1+(i-1)*(ASSIMILATION_WINDOW_LENGTH+1)*2:i*(ASSIMILATION_WINDOW_LENGTH+1)*2) = blockdiag(Q2Day(:,i).',(ASSIMILATION_WINDOW_LENGTH+1)*2); %(2.Day)x(2.Day.Pixels)
    %Q(:,(iN-1)+1:(iN-1)+7)
end  %[Day x (Day . Pixels)]
%end
%end


Htc = [0 0 1 0 -1/2 0]';
for i = 1:ASSIMILATION_WINDOW_LENGTH+1
    %     Ht(:,:,i) = linearObservationOperator(xsf(:,:,i),y(:,:,i),t(:,:,i)); %[6 x Pixels x Day]
    Ht(:,:,i) = [linearObservationOperator2(xsf(:,:,i),y(:,:,i),t(:,:,i)); Htc*ones(1,size(y,1)*size(y,2))];%[(6.2) x Pixels x Days]
end



%[length(find(isnan(Ht(:))==1)) length(find(isnan(xsf(:))==1)) length(find(isnan(x_(:))==1))]

%Hb = (blockdiag(Ht,N+1)).';
for i = 1:size(Ht,2) %[Pixels]
    %Hblock(1:ASSIMILATION_WINDOW_LENGTH+1,1+(i-1)*((ASSIMILATION_WINDOW_LENGTH+1)*7):i*((ASSIMILATION_WINDOW_LENGTH+1)*7)) = blockdiag(squeeze(Ht(:,i,:)),ASSIMILATION_WINDOW_LENGTH+1).';
    Hblock(1:(ASSIMILATION_WINDOW_LENGTH+1)*2,1+(i-1)*((ASSIMILATION_WINDOW_LENGTH+1)*6):i*((ASSIMILATION_WINDOW_LENGTH+1)*6)) = blockdiag(reshape(squeeze(Ht(:,i,:)),6,2*(ASSIMILATION_WINDOW_LENGTH+1)),ASSIMILATION_WINDOW_LENGTH+1).';
end %[(Days.2) x (Days . 6 . Pixels)]



KeepFireDetectx = reshape(KeepFireDetect(:,:,timestamp),1,size(KeepFireDetect,1)*size(KeepFireDetect,2));

for i = 1:size(Ht,2) %Find xa per pixel

    if KeepFireDetectx(i)== 0

        xcontrol = []; %x = [];
        j = 1;
        %t = [];
        %w is unknown, execute Step 2 and Step 3:
        wt = precsolverStep_v5(F, reshape(squeeze(Ht(:,i,:)),6,2*(ASSIMILATION_WINDOW_LENGTH+1)), Hblock(1:(ASSIMILATION_WINDOW_LENGTH+1)*2,1+(i-1)*((ASSIMILATION_WINDOW_LENGTH+1)*6):i*((ASSIMILATION_WINDOW_LENGTH+1)*6)), Q1(1:6,1+(i-1)*6:i*6), Bo(1:6,1+(i-1)*6:i*6), ASSIMILATION_WINDOW_LENGTH, Q2Day(:,i).', Q2Block(1:(ASSIMILATION_WINDOW_LENGTH+1)*2,1+(i-1)*(ASSIMILATION_WINDOW_LENGTH+1)*2:i*(ASSIMILATION_WINDOW_LENGTH+1)*2), yoblock(:,i), xsfvector(:,i),j,xcontrol,tblock(:,i),tHorizonblock(:,i));
        %wt = [rand rand rand rand rand rand]; %It includes w0
        at(:,:,j) = wt;

        %Having w, execute Step 4 and Step 5
        [xa g] = postMultiplier_v5(F,reshape(squeeze(Ht(:,i,:)),6,2*(ASSIMILATION_WINDOW_LENGTH+1)),wt,Q1(1:6,1+(i-1)*6:i*6),Bo(1:6,1+(i-1)*6:i*6),ASSIMILATION_WINDOW_LENGTH,squeeze(xsf(:,i,:)));



        %for $0 \leq n \leq N$.
        %x
        %x(:,0+1,j-1) = x;
        xcontrol(1:size(xa,1),0+1:ASSIMILATION_WINDOW_LENGTH+1,1) = xa; %0+1: time step, 1:iteration
        %6 x Day x Iteration(loop number)



        %for j>1
        %-------
        %-------
        %Outer loop
        %deltaxp(:,n,0) = xp(:,n,0) - x(:,n,-1) for j=1

        %Outer loop (can also use Picard iteration)
        %-----------

        dxa = 0 + g;

        %while j<30
        xcontrolj = xcontrol(:,:,j);
        while (norm(dxa(:))/norm(xcontrolj(:))) >=0.001
            %while (norm(dxa(:),inf)/norm(xcontrolj(:),inf)) >=0.001
            %while norm(dxa(:)/norm(xcontrolj(:))) >=0.001
            %while norm(dxa(:)/norm(xcontrolj(:))) >=0.001
            %while norm(dxa(:)/xcontrolj(:))>=0.001
            %while (norm(dxa)/norm(xcontrolj)) >=0.001


            j=j+1;
            %Step 1:
            %     temp = F;
            %     clear F
            %     F(:,:,1) = temp;
            %     F(:,:,j) = temp;

            if j==2
                deltaxp = zeros(size(x_i,1),ASSIMILATION_WINDOW_LENGTH+1,j);
            else
                deltaxp(:,:,j) = zeros(size(x_i,1),ASSIMILATION_WINDOW_LENGTH+1,1)
            end
            %deltaxp(:,0+1,j) = xb - x(:,0+1,j-1);
            %deltaxp(:,0+1,j) = xb - xcontrol(:,0+1,j-1);
            deltaxp(:,0+1,j) = x_i(:,i) - xcontrol(:,0+1,j-1);

            for k = 1+1:ASSIMILATION_WINDOW_LENGTH+1
                %deltaxp(:,n,j) -  M(:,n-1,j-1)*deltaxp(:,n-1,j) = 0
                %deltaxp(:,n,j) =  F(:,n-1,j-1)*deltaxp(:,n-1,j);
                %deltaxp(:,k,j) =  F(:,(k-1)*size(F,1)+1:k*size(F,1),j)*deltaxp(:,k-1,j);
                deltaxp(:,k,j) =  F*deltaxp(:,k-1,j);
                %deltaxp(:,n,j) =  F(:,(i-1)*size(F,1)+1:i*size(F,1),j)*deltaxp(:,n-1,j)+Q(:,n-1)*q(:,n);

                %xsf = [xsf modelForecast(xsf(:,end))];
            end
            deltaxpvector = (reshape(deltaxp(:,:,j),1,size(deltaxp,1)*size(deltaxp,2))).';

            %Inner loop
            %-----------
            %Step 2 and Step 3:
            %a^j is unknown
            %---
            %at(:,:,j) = wt;
            %---
            %a(:,:,2) = [rand rand rand rand rand rand]; %It includes w0
            %wt = [rand rand rand rand rand rand]; %It includes w0
            %wb = wt.';




            if j==2
                Ht_periteration(:,:,1) = squeeze(Ht(:,i,:));
            end
            %instead of y(:,:,i) use yb, but in case y is of dimension greater than
            %1, this line has to be changed again
            for k = 1:ASSIMILATION_WINDOW_LENGTH+1
                %             Ht_periteration(:,k,j) = [linearObservationOperator2(xcontrol(:,k,j-1),yblock(k,i),tblock(k,i));
                %                        Htc*ones(1,size(y,1)*size(y,2))];%[(6.2) x Pixels x Day]
                Ht_periteration(:,k,j) = [linearObservationOperator2(xcontrol(:,k,j-1),yblock(k,i),tblock(k,i));
                    Htc];%[(6.2) x Pixels x Day]
            end  %[(6.2) x Day x iteration]
            %end


            if j==2
                Hb_periteration(:,:,1) = Hblock(1:(ASSIMILATION_WINDOW_LENGTH+1)*2,1+(i-1)*((ASSIMILATION_WINDOW_LENGTH+1)*6):i*((ASSIMILATION_WINDOW_LENGTH+1)*6)); %Hblock(:,i,:);
            end
            %Hblock(1:ASSIMILATION_WINDOW_LENGTH+1,1+(i-1)*((ASSIMILATION_WINDOW_LENGTH+1)*7):i*((ASSIMILATION_WINDOW_LENGTH+1)*7)) = ;
            Hb_periteration(:,:,j) = blockdiag(reshape(squeeze(Ht_periteration(:,:,j)),6,2*(ASSIMILATION_WINDOW_LENGTH+1)),ASSIMILATION_WINDOW_LENGTH+1).';
            %(2.Days) x (6.Days) x iterations
            %end

            %     temp1 = Ht;
            %     temp2 = Hb;
            %     clear Ht
            %     clear Hb
            %     Ht(:,:,1) = temp1;
            %     Hb(:,:,1) = temp2;
            %     Ht(:,:,j) = [rand(1,7)' rand(1,7)' rand(1,7)' rand(1,7)' rand(1,7)' rand(1,7)']; %Ht: Transposition of each block
            %     Hb(:,:,j) = (blockdiag(Ht(:,:,j),N+1)).';
            %
            %     t = 12:0.15:13.15;
            %,y,
            %a is unknown, execute Step 2 and Step 3:
            %ab = solverStep(F(:,:,j), Ht(:,:,j), Hb(:,:,j-1), Q, Bo, N, R, Rb, yb, deltaxpvector,j,x,t);
            %wt = [rand rand rand rand rand rand]; %It includes w0
            %a(:,:,2) = [rand rand rand rand rand rand]; %It includes w0
            %at(:,:,2) = precsolverStep(F(:,:,j), Ht(:,:,j-1), Hb(:,:,j-1), Q, Bo, N, R, Rb, yb, deltaxpvector,j,x,t); %It includes w0
            at(:,:,j) = precsolverStep_v5(F, reshape(Ht_periteration(:,:,j-1),6,2*(ASSIMILATION_WINDOW_LENGTH+1)), Hb_periteration(:,:,j-1), Q1(1:6,1+(i-1)*6:i*6), Bo(1:6,1+(i-1)*6:i*6), ASSIMILATION_WINDOW_LENGTH, Q2Day(:,i).', Q2Block(1:(ASSIMILATION_WINDOW_LENGTH+1)*2,1+(i-1)*(ASSIMILATION_WINDOW_LENGTH+1)*2:i*(ASSIMILATION_WINDOW_LENGTH+1)*2), yoblock(:,i), deltaxpvector,j,xcontrol,tblock(:,i),tHorizonblock(:,i)); %It includes w0
            %(Hb(:,:,j-1)*Pb(:,:,j-1)*Hb(:,:,j-1).' + Rb)*a(:,:,j) = yb - hnlb - Hb(:,:,j-1)*deltaxp(:,:,j)
            %(Hb(:,:,j-1)*Pb(:,:,j-1)*Hb(:,:,j-1).' + Rb)*a(:,:,j)
            %sqrt(inv(R))*(Hb(:,:,j-1)*Pb(:,:,j-1)*Hb(:,:,j-1).')*ab

            % %block pb
            % pb + sqrt(Rb)*ab
            %
            % for i = 0+1:N+1
            %     hnl(:,i) = observationFunction(x(:,i,j-1),y,t);
            % end
            % hnlb = hnl.';
            % %yb - hnlb - Hb(:,:,j-1)*deltaxpb(:,:,j)
            % sqrt(inv(Rb))*(yb - hnlb - Hb(:,:,j-1)*deltaxpvector)
            % %sqrt(inv(Rb))*(yb-Hb*xsfvector)






            %Having a, execute Step 4 and Step 5
            %xa = postMultiplier(F(:,:,j),Ht(:,:,j-1),at(:,:,2),Q,Bo,N,deltaxp(:,:,j));
            %xa = postMultiplier(F(:,:,j),Ht(:,:,j-1),at(:,:,2),Q,Bo,N,deltaxp(:,:,j));
            [dxa g] = postMultiplier_v5(F,reshape(Ht_periteration(:,:,j-1),6,2*(ASSIMILATION_WINDOW_LENGTH+1)),at(:,:,j),Q1(1:6,1+(i-1)*6:i*6),Bo(1:6,1+(i-1)*6:i*6),ASSIMILATION_WINDOW_LENGTH,deltaxp(:,:,j));


            %for $0 \leq n \leq N$.
            %x
            %x(:,0+1,j-1) = x;
            xcontrol(1:size(xa,1),0+1:ASSIMILATION_WINDOW_LENGTH+1,j) = xcontrol(:,:,j-1) + dxa; %0+1: time step, 1:iteration


            %     if norm(g)<0.01 or norm(dxa-deltaxpvector)<0.01
            %         break;
            %     end

            %if norm(g)<0.1
            %if j>=30
            if j>=(ASSIMILATION_WINDOW_LENGTH+1)
                break;
            end

            xcontrolj = xcontrol(:,:,j);
        end
        xs_i(:,i) = xcontrol(:,1,j); %Analysis estimate
        xs_f(:,i) = xcontrol(:,ASSIMILATION_WINDOW_LENGTH+1,j); %Analysis estimate

        %SCONTROL = size(xcontrol)
        %KeepJJ = [KeepJJ [j norm(g)].'];

        %     if KeepFireDetect(:,:,timestamp)== 1
    else
        xs_i(:,i) = xsf(:,1,1);
        xs_f(:,i) = xsf(:,1,ASSIMILATION_WINDOW_LENGTH+1);
    end
    %     xcccc = size(xcontrol)
    %     xsfff = size(xsf)
    %     xsfvv = size(xsfvector)


    %xs(:,i) = xcontrol(:,1,j); %Analysis estimate



    % Pbb
    % deltaxpb
    % ablock
    %
    % %Having a^j
    % b = backwardSweep(F,Ht,wt,N);
    % g = forwardSweep(F,Q,Bo,b,N);
    %
    %
    % b = backwardSweep(F,H,a,N);
    % g = forwardSweep(F,Q,Pb,b,N);



end


%[length(find(isnan(xa(:))==1)) length(find(isnan(at(:))==1)) length(Ht_periteration(:)==1) length(Hb_periteration(:)==1) length(find(isnan(deltaxpvector)==1)) length(find(isnan(xcontrol(:,:,1))==1)) length(find(isnan(xcontrol(:,:,2))==1)) length(find(isnan(xa)==1)) length(find(isnan(wt)==1)) length(find(isnan(Ht)==1)) length(find(isnan(Q2Day)==1)) length(find(isnan(xsfvector)==1)) length(find(isnan(inn)==1))]

%[length(find(isnan(Ht_periteration(:,:,1))==1)) length(find(isnan(Ht_periteration(:,:,2))==1)) length(find(isnan(xs)==1)) length(find(isnan(x_)==1)) length(find(isnan(xcontrol(:,:,1))==1)) length(find(isnan(xcontrol(:,:,2))==1)) length(find(isnan(deltaxpvector)==1)) length(find(isnan(at(:,:,1))==1)) length(find(isnan(at(:,:,2))==1))]

%j
%size(xcontrol)
%size(at)
%[xcontrol(:,1,1).' length(find(isnan(xcontrol(:,:,2))==1)) length(find(isnan(at(:,:,2))==1))]


%ESTIMATE Kn

%xs = x_(:,:,timestamp) + Gf.*alpha; %Filtered  estimate while x_(n) is predicted estimate
%xs = x_ + extend(Gf,Ne).*alpha; %Filtered  estimate while x_(n) is predicted estimate
% xs = x_ + incremental(Gf,alpha); %Filtered  estimate while x_(n) is predicted estimate
%ESTIMATE Kn


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

Keepxs(:,:,timestamp) = xs_f;


x_i = xs_i; %+ process_noise; [No noise is added]

%xxxx_ = size(x_i)
x_f = xs_f;

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
% EnsPXs = xs - extend(mean(xs,3),Ne);
% %Kn_(1:size(t,1),1:size(t,2),timestamp) = calculateForecastErrorCovariance(EnsPX_);
% Kn = calculateForecastErrorCovariance(EnsPXs);



%Kn_(1:size(t,1),1:size(t,2),timestamp+1) = F.* Kn .*F + Q1; %Correlation matrix of error in the predicted estimate for the next sample
% EnsPX_ = x_ - extend(mean(x_,3),Ne);
% %Kn_(1:size(t,1),1:size(t,2),timestamp) = calculateForecastErrorCovariance(EnsPX_);
% Kn_ = calculateForecastErrorCovariance(EnsPX_);

%prediction(:,:,timestamp+1) = mean(x_,3);
prediction(:,:,timestamp+1) = x_f;


% if sum(isnan(x_))>0
%     disp('x_ has Nan')
% end
% if sum(isnan(prediction(:,:,timestamp+1)))>0
%     disp('prediction has Nan')
% end
%predictedObservation(:,:,timestamp+1) = observationFunction(prediction(:,:,timestamp+1),y(:,:,1),t);
predictedObservation(:,:,timestamp+1) = observationFunction(prediction(:,:,timestamp+1),y(:,:,ASSIMILATION_WINDOW_LENGTH+1),t(:,:,ASSIMILATION_WINDOW_LENGTH+2));
if timestamp == ASSIMILATION_WINDOW_LENGTH + 1
    te = t(:,:,1:ASSIMILATION_WINDOW_LENGTH+1);
    ye = y(:,:,1:ASSIMILATION_WINDOW_LENGTH+1);
    %he = Ht(:,:,1:ASSIMILATION_WINDOW_LENGTH+1);
    he = Ht(:,:,ASSIMILATION_WINDOW_LENGTH+1);
else
    te = t(:,:,ASSIMILATION_WINDOW_LENGTH+1);
    ye = y(:,:,ASSIMILATION_WINDOW_LENGTH+1);
    he = Ht(:,:,ASSIMILATION_WINDOW_LENGTH+1);
end
%Rea = Q2Day(1:2,:); %R1a, R for day 1
Rea = Q2Day(end-1:end,:); %R1a, R for day 1

%     KKK = Kn(n)
%     KKK_ = Kn_(n)
%     SSS = S(n)

% S(S==0) = eps; %For S= 0, I replaced it with a small value
% stResidual = inn./sqrt(S); %Standardized residual

