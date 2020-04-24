function [KeepFireDetect x_ prediction Keep_Q Keep_Inn Keep_S stResidual z cycle_out_scope cycle_reported estimation Keep_R] = stdsirDECONV10(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,prediction,Keep_Q,Keep_Inn,Keep_S,KeepFireDetect,DECONV,restored,Keep_R,NUMBER_PARTICLE_SIR)
%function [KeepFireDetect x_ prediction Keep_Q Keep_Inn Keep_S stResidual z cycle_out_scope cycle_reported estimation Keep_R] = stdsirDECONV6(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,prediction,Keep_Q,Keep_Inn,Keep_S,KeepFireDetect,DECONV,restored,Keep_R)
%function [KeepFireDetect x_ prediction Keep_Q Keep_Inn Keep_S stResidual z cycle_out_scope cycle_reported estimation Keep_R] = stdsirDECONV6(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,prediction,Keep_Q,Keep_Inn,Keep_S,KeepFireDetect,DECONV,restored,Keep_R)
%function [x_ Keep_Q Keep_Inn stResidual Keep_R] = stdsirDECONV5(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,Keep_Q,Keep_Inn,DECONV,restored,Keep_R)
%function [KeepFireDetect x_ Keep_Q Keep_Inn stResidual Keep_R] = stdsirDECONV5(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,Keep_Q,Keep_Inn,KeepFireDetect,DECONV,restored,Keep_R)
%function [KeepFireDetect x_ prediction Keep_Q Keep_Inn Keep_S stResidual z cycle_out_scope cycle_reported estimation Keep_R] = stdsirDECONV4(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,prediction,Keep_Q,Keep_Inn,Keep_S,KeepFireDetect,DECONV,restored,Keep_R)
%function [time_  cycle_out_scope stResidual]= stdsir_ALTDECONV(v4,w,TIME,CURRENT_CYCLE,latitude,longitude,column,row,NUMBER_STANDARD_DEVIATION,To_M, Ta_M, tm_M, ts_M, dT_M, Q, R1, bA, MNs, mp, cycle_out_scope)
%function [time_  cycle_out_scope stResidual]= stdsirDECONV    (v4,w,TIME,CURRENT_CYCLE,latitude,longitude,column,row,NUMBER_STANDARD_DEVIATION,To_M, Ta_M, tm_M, ts_M, dT_M, Q, R1, bA, MNs, mp, cycle_out_scope)


%sa = randn('state');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%randn('state',0);
%Keep_S = [];
% sb = rand('state');
% rand('state',0);

%M = 1; %Dimension of the state variable
%N = 1; %Dimension of the observation


cycle_reported = [];


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
Q = zeros(NUMBER_ROW,NUMBER_COLUMN);
R1 = zeros(NUMBER_ROW,NUMBER_COLUMN);
cycle_out_scope = zeros(NUMBER_ROW,NUMBER_COLUMN);
if timestamp==1
    Qi = zeros(NUMBER_ROW,NUMBER_COLUMN);
end


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
        %if timestamp==1
        %Qi(jN,iN) = TESTINGCYCLE_DATA(jN,iN).Q(n(jN,iN));
        %n = n + 1;
        %Q(jN,iN) = TESTINGCYCLE_DATA(jN,iN).Q(n(jN,iN));
        %n = n - 1;
        %else
        %    n = n + 1;
        Q(jN,iN) = TESTINGCYCLE_DATA(jN,iN).Q(n(jN,iN));
        %n = n - 1;
        %end
        R1(jN,iN) = TESTINGCYCLE_DATA(jN,iN).R1(n(jN,iN));
        cycle_out_scope(jN,iN) = TESTINGCYCLE_DATA(jN,iN).TREATED(n(jN,iN));
    end
end

Ns = NUMBER_PARTICLE_SIR; %500;%450;%800; %Number of particles


if DECONV==1
    z = restored;
else
    z = v4; %I can optimize the code by not reading v4 in case of DECONV = 1
end
clear v4 restored

%Q1 = Q; %Process noise covariance

xf = zeros(NUMBER_ROW,NUMBER_COLUMN,Ns); %xf is filtered samples, it will keep all filtered distribution, each distribution on a each pixels


%x_ = [];
%z = v4;  %Observation








% Keep_Q(1:size(t,1),1:size(t,2),timestamp) = Q;
% if  length(find(isnan(Q)==1))~=0
%     windowSize_row = size(Q,1);
%     windowSize_column = size(Q,2);
%     window_start_row = 1;
%     window_start_column = 1;
%     %         window_centre_row = ceil(windowSize_row/2); %Case the window size is an even number? What will happen?
%     %         window_centre_column = ceil(windowSize_column/2); %Case the window size is an even number? What will happen?
% 
%     SRQ = restoreMissingSamples10(sqrt(Q),timestamp,windowSize_row,windowSize_column,window_start_row, window_start_column,sqrt(Keep_Q));
%     SRQ(SRQ<0) = 0;
%     Q = SRQ.^2;
% end
% clear SRQ
% 
% Keep_Q(1:size(t,1),1:size(t,2),timestamp) = Q;

%Q = extend(Q,Ns); %Extend a 2D-matrix to a 3D-matrix by repeating the 2D-matrix content
process_noise = extend(sqrt(Q),Ns).*randn(size(xf));    %Process noise assumed to be Gaussian distributed (process noise is a distribution while Q(Q1) is a point)
%process_noise = extend(sqrt(Q)).*randn(size(xf));    %Process noise assumed to be Gaussian distributed
%n(:,t) = sqrt(Q(t))*randn(size(xf(:,t)));    %Process noise assumed to be Gaussian distributed
%n(:%,t) = normrnd(0,sqrt(Q(t)),size(xf(:,t)));

if timestamp==1
    x_= extend(To+Ta.*cos(pi./w1.*(t-tm)),Ns) + process_noise; %Initial predicted state estimate distribution %To+n(:,1)%z(1)+n(:,1)%To+Ta*cos(pi/w*(ct-tm))+n(:,1);%z(1)+n(:,1);%%######################CHECK
    clear Qi
end

%Enter the algorithm with a prediction of current time
%m = x_;%(x_(:,t).^(2))./20;
%prediction(:,:,timestamp) = mean(x_,3); %THE FIRST PREDICTION: for standard SIR (equal weight, so as to use mean)
%Q2 = R1;


R = R1; %Measurement noise covariance
Keep_R(1:size(t,1),1:size(t,2),timestamp) = R; %Keep the measurement covariance
if  length(find(isnan(R)==1))~=0
    windowSize_row = size(R,1);
    windowSize_column = size(R,2);
    window_start_row = 1;
    window_start_column = 1;
    %     window_centre_row = ceil(windowSize_row/2); %Case the window size is an even number? What will happen?
    %     window_centre_column = ceil(windowSize_column/2); %Case the window size is an even number? What will happen?

    %Kn_(:,:,timestamp) = restoreMissingSamples6(Kn_(:,:,timestamp),timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,Kn_);

    %Kn_(:,:,timestamp) = LaplacianInterpolation(Kn_(:,:,timestamp));
    %Two methods, one using Kn_ and another one using S
    %Kn_(:,:,timestamp) = restoreMissingSamples6(Kn_(:,:,timestamp),timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,Kn_);

% % % % % % % % % % % %     SR_R = restoreMissingSamples10(sqrt(R),timestamp,windowSize_row,windowSize_column,window_start_row, window_start_column,sqrt(Keep_R));
% % % % % % % % % % % %     SR_R(SR_R<0) = 0;
% % % % % % % % % % % %     R = SR_R.^2;
    %S = restoreMissingSamples6(S,timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,Keep_S);
    %S = Kn_(:,:,timestamp) + Q2;
    %Kn_(:,:,timestamp) = S - Q2; %Considering that C is a scalar equal to 1
end
clear SR_R
%R = Q2;
%R1 = Q2;
%S = Kn_(:,:,timestamp) + Q2;


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %FIRE DETECTION FOR PAPER 2.

prediction(:,:,timestamp) = mean(x_,3); %THE FIRST PREDICTION: for standard SIR (equal weight, so as to use mean)
%prediction = mean(x_,3); %THE FIRST PREDICTION: for standard SIR (equal weight, so as to use mean)
inn = z - prediction(:,:,timestamp); %Inn is the residual
%inn = z - prediction; %Inn is the residual
S = mean((x_ - extend(prediction(:,:,timestamp),Ns)).^2,3) + R; %Q2=R(missing values removed)Residual covariance: for standard SIR (equal weight, so as to use mean)
%S = mean((x_ - extend(prediction,Ns)).^2,3) + R; %Q2=R(missing values removed)Residual covariance: for standard SIR (equal weight, so as to use mean)
%Keep_Inn(1:size(t,1),1:size(t,2),timestamp) = inn;
%Keep_S(1:size(t,1),1:size(t,2),timestamp) = S;

% if length(find(isnan(inn)==1))~=0
%     Keep_Inn(1:size(t,1),1:size(t,2),timestamp) = inn;
%     for j=1:Ns
%         sampleInn = z - x_(:,:,j);
%         windowSize_row = size(sampleInn,1);
%         windowSize_column = size(sampleInn,2);
%         window_start_row = 1;
%         window_start_column = 1;
%         %         window_centre_row = ceil(windowSize_row/2); %Case the window size is an even number? What will happen?
%         %         window_centre_column = ceil(windowSize_column/2); %Case the window size is an even number? What will happen?
% 
%         sampleInn = restoreMissingSamples10(sampleInn,timestamp,windowSize_row,windowSize_column,window_start_row, window_start_column,Keep_Inn); %Keep_Inn for the residual not over MC samples
%         %alpha=LaplacianInterpolation(alpha);
%         %Cn1 = z - inn;
%         %x_(:,:,timestamp) = Cn1;
%         x_(:,:,j) = z - sampleInn;
%     end
%     clear sampleInn;
%     prediction(:,:,timestamp) = mean(x_,3); %THE FIRST PREDICTION: for standard SIR (equal weight, so as to use mean)
%     %prediction = mean(x_,3); %THE FIRST PREDICTION: for standard SIR (equal weight, so as to use mean)
%     inn = z - prediction(:,:,timestamp); %Inn is the residual
%     %inn = z - prediction; %Inn is the residual
%     S = mean((x_ - extend(prediction(:,:,timestamp),Ns)).^2,3) + R; %Q2=R(missing values removed)Residual covariance: for standard SIR (equal weight, so as to use mean)
%     %S = mean((x_ - extend(prediction,Ns)).^2,3) + R; %Q2=R(missing values removed)Residual covariance: for standard SIR (equal weight, so as to use mean
% end
%clear prediction


% tempKn_(1:size(Kn_,1),1:size(Kn_,2)) = Kn_(:,:,timestamp);
% if  length(find(isnan(tempKn_)==1))~=0
%     windowSize_row = size(tempKn_,1);
%     windowSize_column = size(tempKn_,2);
%     window_centre_row = ceil(windowSize_row/2); %Case the window size is an even number? What will happen?
%     window_centre_column = ceil(windowSize_column/2); %Case the window size is an even number? What will happen?
%
%     SRKn_ = restoreMissingSamples6(sqrt(Kn_(:,:,timestamp)),timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,sqrt(Kn_));
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


% if  length(find(isnan(Q2)==1))~=0
%     windowSize_row = size(Q2,1);
%     windowSize_column = size(Q2,2);
%     window_centre_row = ceil(windowSize_row/2); %Case the window size is an even number? What will happen?
%     window_centre_column = ceil(windowSize_column/2); %Case the window size is an even number? What will happen?
%
%     %Kn_(:,:,timestamp) = restoreMissingSamples6(Kn_(:,:,timestamp),timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,Kn_);
%
%     %Kn_(:,:,timestamp) = LaplacianInterpolation(Kn_(:,:,timestamp));
%     %Two methods, one using Kn_ and another one using S
%     %Kn_(:,:,timestamp) = restoreMissingSamples6(Kn_(:,:,timestamp),timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,Kn_);
%
%     SRQ2 = restoreMissingSamples6(sqrt(Q2),timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,sqrt(Keep_Q2));
%     SRQ2(SRQ2<0) = 0;
%     Q2 = SRQ2.^2;
%     %S = restoreMissingSamples6(S,timestamp,windowSize_row,windowSize_column,window_centre_row, window_centre_column,Keep_S);
%     %S = Kn_(:,:,timestamp) + Q2;
%     %Kn_(:,:,timestamp) = S - Q2; %Considering that C is a scalar equal to 1
% end
% S = Kn_(:,:,timestamp) + Q2;


% %Keep_Q(1:size(t,1),1:size(t,2),timestamp+1) = Q;
% Keep_Q(1:size(t,1),1:size(t,2),timestamp) = Q;
% if  length(find(isnan(Q)==1))~=0
%     windowSize_row = size(Q,1);
%     windowSize_column = size(Q,2);
%     window_start_row = 1;
%     window_start_column = 1;
% %     window_centre_row = ceil(windowSize_row/2); %Case the window size is an even number? What will happen?
% %     window_centre_column = ceil(windowSize_column/2); %Case the window size is an even number? What will happen?
%
%     SRQ = restoreMissingSamples10(sqrt(Q),timestamp,windowSize_row,windowSize_column,window_start_row, window_start_column,sqrt(Keep_Q));
%     SRQ(SRQ<0) = 0;
%     Q = SRQ.^2;
% end
% clear SRQ
% %Q = Q1;
% process_noise = extend(sqrt(Q),Ns).*randn(size(xf));    %Process noise assumed to be Gaussian distributed (process noise is a distribution while Q(Q1) is a point)
%


Keep_Inn(1:size(t,1),1:size(t,2),timestamp) = inn;
Keep_S(1:size(t,1),1:size(t,2),timestamp) = S; %Comment to save space
Keep_R(1:size(t,1),1:size(t,2),timestamp) = R; %Keep the measurement covariance
%Keep_Q(1:size(t,1),1:size(t,2),timestamp) = Q;
clear Q


% m = x_(:,t+1);
%
% prediction(t+1) = mean(x_(:,t+1)); %prediction estimate for standard SIR (equal weight, so as to use mean)
% %prediction(t+1) = x_(ixp(1),t+1);
% %     bins = 20;
% %     [p,pos]=hist(x_(:,t+1),bins);
% %     map=find(p==max(p));
% %     prediction(t+1)=pos(map(1));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %FIRE DETECTION FOR PAPER 2.
% inn(t+1) = z(t+1) - prediction(t+1);
% S(t+1) = mean((m - prediction(t+1)).^2) + R; %Residual covariance: for standard SIR (equal weight, so as to use mean)
%




%x_(1:size(t,1),1:size(t,2),timestamp) = To+Ta.*cos(pi./w.*(t-tm)); %xs; %TO CHECK
%Kn_(1:size(t,1),1:size(t,2),timestamp) = Q; %10;%xcorr



%     sigma = range(x_(:,t))/Ns
%     sigma = sqrt(R);
%     for s=1:Ns
%         itg(s) = quad(@(theta) gk(theta,x_(s,t),sigma),-Inf,z(t));
%     end
%     ur(t) = sum(itg)/Ns;


%REMOVED
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %sigma = range(x_(:,t))/Ns;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % sigma = sqrt(S); %sqrt(S(t)) or  sqrt(S(t+1)); %sqrt(R);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % for s=1:Ns
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %itg(s) = 1/2*(1 + erf((z(t) - x_(s,t))/sqrt(2*sigma^2)));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %itg(s) = 1 - 1/2*erfc((z(t) - x_(s,t))/sqrt(2*sigma^2));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     itg(:,:,s) = 1 - 1/2*erfc((z - x_(:,:,s))./sqrt(2*sigma.^2));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % ur = sum(itg,3)/Ns;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % vr = norminv(ur,0,1);



% for s=1:Ns  --t not t+1
%     %itg(s) = 1/2*(1 + erf((z(t+1) - x_(s,t+1))/sqrt(2*sigma^2)));
%     itg(s) = 1 - 1/2*erfc((z(t) - x_(s,t))/sqrt(2*sigma^2));
% end
% ur(t+1) = sum(itg)/Ns;
%
% vr(t+1) = norminv(ur(t+1),0,1);
%
%
% for s=1:Ns
%     %itg(s) = 1/2*(1 + erf((z(t+1) - x_(s,t+1))/sqrt(2*sigma^2)));
%     itg(s) = 1 - 1/2*erfc((z(t+1) - x_(s,t+1))/sqrt(2*sigma^2));
% end
% ur(t+1) = sum(itg)/Ns;
%
% vr(t+1) = norminv(ur(t+1),0,1);



%keepvr(:,:,timestamp) = vr;


%sigma = range(x_(:,t+1))/Ns;
% sigma = sqrt(R);%sqrt(S(t+1));
%
% for s=1:Ns
%     %itg(s) = 1/2*(1 + erf((z(t+1) - x_(s,t+1))/sqrt(2*sigma^2)));
%     itg(s) = 1 - 1/2*erfc((z(t+1) - x_(s,t+1))/sqrt(2*sigma^2));
% end
% ur(t+1) = sum(itg)/Ns;
%
% vr(t+1) = norminv(ur(t+1),0,1);



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % K = 1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % D = (MNs(1:1,CYCLE) + MNs(1:1,CYCLE-1))/2;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Mn = mean(D);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Sn = std(D);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Tc = Mn + K*Sn;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %Tc = K;


% Tc = NUMBER_STANDARD_DEVIATION;
%
% %R = R1;
% if (vr(t+1)>Tc)%(inn(t+1)>Tc)
%
%     fire_detect(t+1) = 1;
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             %R2 = R1;
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             R2 = inn(t+1)^2;
%     %             R = inn(t+1)^2;
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             %prediction(t+1) = x_(ixp(1),t+1);
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             %R2 = cov(z(t+1)-x_(:,t+1));
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             checkTc = [checkTc Tc];
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             checkinn = [checkinn inn(t+1)];
%     %             R = 1/12*(335 - prediction(t+1) - Tc)^2; %Assuming a uniform distribution from Tc to (335 - prediction)
%     %Change only likelihood = change only R (increase this variance or make R = Inf (R=Inf, resampling is not important) so that the prior can have a priority)
%     %residual covariance can be changed inside here, but to keep it as predicted it is not implemented inside here
%
%     R = Inf;
%
% else
%     fire_detect(t+1) = 0;
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             R2 = R1; %MUST ALSO CHANGE ABOVE IF THERE IS  A CHANGE HERE, ON THE MEASUREMENT VARIANCE
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             %R2 = inn(t+1)^2;
%     %Change only likelihood = change only R
%     %residual covariance can be changed inside here, but to keep it as predicted it is not implemented inside here
%     R = R1;
% end


%Tc = sqrt(S(t))*0.5;%T1;
Tc = sqrt(S).*NUMBER_STANDARD_DEVIATION; %Tc = sqrt(S(t+1))*NUMBER_STANDARD_DEVIATION;%100; Tc = 0.5
%Tc = NUMBER_STANDARD_DEVIATION;
%Tc = NUMBER_STANDARD_DEVIATION;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

%Tc = 100;
%fire_detect = alpha > Tc;
fire_detect = inn > Tc;
KeepFireDetect(1:size(t,1),1:size(t,2),timestamp) = fire_detect;
R = R1;
R(fire_detect==1) = Inf;
%clear Tc

%read_fire_detect = fire_detect;
%Gf = mod(fire_detect + 1,2); %CASE OF FIRE
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % R = R1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Q2 = R;
% S = C.*Kn_(:,:,timestamp).*C'+Q2; %This line and the following
% S(S==0) = eps;
% Gf = mod(fire_detect + 1,2).*Kn_(:,:,timestamp).*C'.*(S).^(-1); %Done two times, abuse CASE OF NO FIRE (Fire=1, no fire = 0)



% if (vr(t)>Tc)%(inn(t)>Tc)
%
%     fire_detect(t) = 1;
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %R2 = R1;
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %         R2 = inn(1)^2;
%     %         R = inn(t)^2;
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %         Update the current filtered estimate distribution xf(:,1)
%     %         R = 1/12*(335 - prediction(t) - Tc)^2; %Assuming a uniform distribution from Tc to (335 - prediction)
%     R = Inf;
% else
%     fire_detect(t) = 0;
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %         R2 = R1;
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %R2 = inn(1)^2;
%     R = R1;
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %         Update the current filtered estimate but we don't have to repeat it xf(:,1) This is the same as just updating the likelihood and then filtered
% end


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %R =R1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % R = R2;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %R2 = R1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % m = (m - mean(m))/std(m) *(std(m)*sqrt(R1/R2)) + mean(m);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % x_(:,1) = m;


%Calculate importance weights and normalize the weights
% for s=1:Ns,
%     alpha = z(t+1).*ones(size(m))-m;
%     %q(s,t+1) = exp(-0.5*R^(-1)*(z(t+1)- m(s,1))^(2))./sum(exp(-0.5*R^(-1)*alpha.^(2)));
%     %sum(exp(-0.5*R^(-1)*alpha.^(2)))
%     q(s,t+1) = exp(-0.5*R^(-1)*(z(t+1)- m(s,1))^(2))/sum(exp(-0.5*R^(-1)*alpha.^(2)));
% end;


%Calculate importance weights and normalize them
z_extended = extend(z,Ns);
%R(R==0) = eps;
R_extended = extend(R,Ns);
alpha = z_extended-x_; %m(=x_) is the prediction state estimate distribution
%SCALING = sum(exp(-0.5*R_extended.^(-1).*alpha.^(2)),3);
LIKELIHOOD_F = exp(-0.5*R_extended.^(-1).*(alpha.^(2)));
SCALING = sum(LIKELIHOOD_F,3);
SCALING(SCALING==0) = eps;
clear alpha z_extended R_extended

for s=1:Ns,
    %     alpha = z_extended-m; %m is the prediction state estimate distribution
    %q(s,t+1) = exp(-0.5*R^(-1)*(z(t+1)- m(s,1))^(2))./sum(exp(-0.5*R^(-1)*alpha.^(2)));
    %sum(exp(-0.5*R^(-1)*alpha.^(2)))
    %     if length(find(sum(exp(-0.5*R_extended.^(-1).*alpha.^(2)),3)==0))>0
    %         disp('some zeros in DENOMINATOR')
    %     end
    %
    %     if length(find((0.5*R_extended.^(-1).*alpha.^(2))>10000)) >0
    %         disp('some zeros in R')
    %     end
    %
    %
    %     if length(find(exp(-0.5*R_extended.^(-1).*alpha.^(2))==0))>0
    %         disp('some zero in DENOMINATOR INTERIOR')
    %     end
    %
    %     if length(find(isinf(alpha(:))==1))>0
    %        disp('some zero in alpha')
    %     end
    %SCALING = sum(exp(-0.5*R_extended.^(-1).*alpha.^(2)),3);
    %SCALING(SCALING==0) = eps;

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     q(:,:,s) = exp(-0.5*R.^(-1)*(z- m(:,:,s)).^(2))./SCALING;
    q(:,:,s) = LIKELIHOOD_F(:,:,s)./SCALING;
    %q(:,:,s) = exp(-0.5*R.^(-1)*(z- m(:,:,s)).^(2))./sum(exp(-0.5*R_extended.^(-1).*alpha.^(2)),3);
end;
clear SCALING LIKELIHOOD_F


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % prediction_Likelihood(t) = sum(q(:,t).*x_(:,t));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % inn_Likelihood(t) = z(t) - prediction_Likelihood(t);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % S_Likelihood(t) = mean((m - prediction_Likelihood(t)).^2) + R; %Residual covariance: for standard SIR (equal weight, so as to use mean)


% prediction_Likelihood = sum(q.*x_,3);
% inn_Likelihood = z - prediction_Likelihood;
% S_Likelihood = mean((m - extend(prediction_Likelihood,Ns)).^2,3) + R; %Residual covariance: for standard SIR (equal weight, so as to use mean)




%LPRED = sum(alpha);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     LPRED = z(1)-prediction(1);

%PARALLEL BUT TAKES LONGER BY A FACTOR OF 3
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %Resampling using systematic resampling
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % tic
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % cdf_q = cumsum(q,3); %Construct CDF
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %i = 1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %u(:,:,1) = rand(NUMBER_ROW,NUMBER_COLUMN)*1/Ns; %Draw a starting point
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % u = rand(NUMBER_ROW,NUMBER_COLUMN)*1/Ns; %Draw a starting point
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % for j = 1: Ns
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     uf = u + 1/Ns*(j - 1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     keepcdf_q = cdf_q;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     keepMin = extend(min(cdf_q,[],3),Ns);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     keepGreaterIndex = find(cdf_q >= extend(uf,Ns));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     keepcdf_q(keepGreaterIndex) =  keepMin(keepGreaterIndex) - keepcdf_q(keepGreaterIndex);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     [MinGreater MinGreaterIndex] = min(keepcdf_q,[],3);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     clear MinGreater
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     [I J] = ind2sub (size(MinGreaterIndex), 1:numel(MinGreaterIndex));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     linInd = sub2ind (size (x_), I, J, MinGreaterIndex(:)');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     xf(:,:,j) = reshape (x_(linInd), size(MinGreaterIndex));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %MinGreater = (keepcdf_q==min(keepcdf_q,[],3));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %[II JJ] = find(keepcdf_q==min(keepcdf_q,[],3));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %[II JJ KK] = ind2sub(size(keepcdf_q),find) [r c t] = ind2sub(size(QQ),find(QQ==8))
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %LL = JJ-size(cdf_q,2)*(floor((JJ-1)/size(cdf_q,2)));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %KK=floor((JJ-1)/size(cdf_q,2)) + 1];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %JJ = LL;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %clear LL;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     for k = 1:NUMBER_ROW*NUMBER_COLUMN
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         i(II(k),JJ(k)) = KK(k);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %size(MinGreaterIndex)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %MinGreaterIndex
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %CurrentSample = false(size(xf));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %CurrentSample(:,:,j) = 1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %xf(:,:,j*ones(NUMBER_ROW,NUMBER_COLUMN)) = x_(:,:,i);  %PP = (keepcdf_q==min(keepcdf_q,[],3)) xf() = x_(PP)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %xf(:,:,j*ones(NUMBER_ROW,NUMBER_COLUMN)) = x_(:,:,MinGreaterIndex);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     q(:,:,j) = 1/Ns;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %Assign parent keepi(j,1) = i;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %j
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % toc
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %xf(:,1) = (To+Ta*cos(pi/w*(ct-tm)))*rand(Ns,1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % clear I J linInd MinGreaterIndex keepGreaterIndex keepMin keepcdf_q cdf_q uf u


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % qq = q;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %Resampling using systematic resampling
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % tic
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % cdf_q = cumsum(q,3); %Construct CDF
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %i = 1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % MinGreaterIndex = ones(NUMBER_ROW,NUMBER_COLUMN);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %u(:,:,1) = rand(NUMBER_ROW,NUMBER_COLUMN)*1/Ns; %Draw a starting point
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % u = rand(NUMBER_ROW,NUMBER_COLUMN)*1/Ns; %Draw a starting point
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % for j = 1: Ns
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     uf = u + 1/Ns*(j - 1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     if j==1
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         [I J] = ind2sub (size(MinGreaterIndex), 1:numel(MinGreaterIndex));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         linInd = sub2ind (size(cdf_q), I, J, MinGreaterIndex(:)');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         keepcdf_q = reshape (cdf_q(linInd), size(MinGreaterIndex));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         Greater = (uf>keepcdf_q);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         Greater(MinGreaterIndex==(Ns-1)) = 0;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     while length(find(Greater==1))>0
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         MinGreaterIndex(Greater==1) = MinGreaterIndex(Greater==1) + 1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %[I J] = ind2sub (size(MinGreaterIndex), 1:numel(MinGreaterIndex));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         linInd = sub2ind (size(cdf_q), I, J, MinGreaterIndex(:)');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         keepcdf_q = reshape (cdf_q(linInd), size(MinGreaterIndex));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         Greater = (uf>keepcdf_q);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         Greater(MinGreaterIndex==(Ns-1)) = 0;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %[I J] = ind2sub (size(MinGreaterIndex), 1:numel(MinGreaterIndex));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %linInd = sub2ind (size(x_), I, J, MinGreaterIndex(:)');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     xf(:,:,j) = reshape (x_(linInd), size(MinGreaterIndex));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     q(:,:,j) = 1/Ns;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %Assign parent keepi(j,1) = i;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % toc
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %xf(:,1) = (To+Ta*cos(pi/w*(ct-tm)))*rand(Ns,1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % clear I J linInd MinGreaterIndex keepcdf_q cdf_q uf u Greater


%q = qq;
% %Resampling using systematic resampling
%tic
for irow = 1:NUMBER_ROW
    for jcolumn = 1:NUMBER_COLUMN
        cdf_q = cumsum(q(irow,jcolumn,:)); cdf_q = cdf_q(:);%Construct CDF
        i = 1;
        u(1) = rand*1/Ns; %Draw a starting point
        for j = 1: Ns
            u(j) = u(1) + 1/Ns*(j - 1);
            while (u(j) > cdf_q(i)) && (i < Ns) %(i<500)
                i = i + 1;
            end
            xf(irow,jcolumn,j) = x_(irow,jcolumn,i);
            q(irow,jcolumn,j) = 1/Ns;
            %Assign parent keepi(j,1) = i;
        end
    end
end
%toc
clear cdf_q u q
%xf(:,1) = (To+Ta*cos(pi/w*(ct-tm)))*rand(Ns,1);


% %Systematic resampling
% cdf_q = cumsum(q(:,t+1));
% i = 1;
% u(i) = rand*1/Ns;
% for j = 1: Ns
%     u(j) = u(1) + 1/Ns*(j - 1);
%     while u(j) > cdf_q(i)%&(i<500)
%         i = i + 1;
%     end
%     xf(j,t+1) = x_(i,t+1);
%     q(j,t+1) = 1/Ns;
%     %Assign parent keepi(j,t+1) = i;
% end


%     u = rand(Ns+1,1);
%     r = -log(u);
%     cdf_u = cumsum(r);
%     i = 1;
%     j = 1;
%
%     while j <= Ns,
%         if (cdf_q(j)*cdf_u(Ns)) > cdf_u(i)
%             xf(i,t+1) = x_(j,t);
%             i = i+1;
%         else
%             j = j+1;
%         end;
%     end;

%%%%%
%MEAN estimate for the filtered estimate
% estimation(t+1) = mean(xf(:,t+1));
% xes = estimation(t+1);



% UPDATE AND PREDICTION STAGES:
% ============================

%MEAN ESTIMATE
%estimation(:,:,timestamp) = mean(xf,3);%%%%%%%%%%%%%CHECK prediction(1) = mean(xf(:,1));
estimation = mean(xf,3);%%%%%%%%%%%%%CHECK prediction(1) = mean(xf(:,1));
%xes = estimation(:,:,timestamp);

%MAP (PEAK) estimate
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         bins = 20;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         [p,pos]=hist(xf(:,t),bins);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         map=find(p==max(p));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         estimation(t)=pos(map(1)); %Estimated state
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         xes = estimation(t);
% stResidual(t+1) = vr(t+1);%inn(t+1)/sqrt(S(t+1)); %Standardized residual using predicted residual covariance



%stResidual(:,:,timestamp) = vr;%inn(t)/sqrt(S(t)); %Standardized residual
S(S==0) = eps;
stResidual = inn./sqrt(S);

%ANOTHER METHOD(CHECK ALSO DOWN)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % if ct<ts,
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     if sin((ct-tm)*pi/(4*w))>0
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         prediction(2) = To+(xes - To).*cos(pi/(w*4)) - sqrt(abs(Ta^2 - (xes - To).^2)) .*sin(pi/(w*4));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     else
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         prediction(2) = To+(xes - To).*cos(pi/(w*4)) + sqrt(abs(Ta^2 - (xes - To).^2)) .*sin(pi/(w*4));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % else
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     prediction(2) = (To+dT)+(xes -To-dT)*exp(-1/4/k);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% xp = abs(estimation(1) - xf(:,1));
% [yp,ixp] = min(xp);
% CHECKESTIMATION = xf(ixp(1),1);

%xp = xf(:,1);
%ixp = find(xp==estimation(1));
%ixp

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % LEST = z(1)-estimation(1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%estimation(1) = mean(xf(:,1));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % %z(10) = z(10) + 20;




%stResidual = [];

%Tc = 5;%THRESHOLD
%T1 = 100;
%time_ = [];



%[Q R1 bA MNs] = noisemodel(current_txtfile);


%N = 96;                % Number of time steps.
% LENGTH_CURRENT_CYCLE = length(v4);
% [To_M Ta_M tm_M ts_M dT_M Q R1 bA MNs mp cycle_out_scope] = noisemodel(current_txt_file,CURRENT_CYCLE,latitude, longitude,LENGTH_CURRENT_CYCLE);

%R1

% To = To_M;
% Ta = Ta_M;
% tm = tm_M;
% ts = ts_M;
% %k = k_M;
% dT = dT_M;
% %w = w_M;
% k = w/pi*((1/tan(pi/w*(ts-tm)))-dT/Ta*(1/sin(pi/w*(ts-tm))));

% dayt = TIME;
% MODEL = (To + Ta *cos(pi/w *(dayt-tm))).*(dayt<ts)+((To + dT)+(Ta *cos(pi/w*(ts -tm))-dT)*exp(-(dayt-ts)/k)).*(dayt>=ts);
%
% figure
% errorbar(dayt,MODEL,sqrt(Q))
% hold on
% plot(dayt,v4,'r*')
%
% %plotyy(dayt,v4,dayt,sqrt(R1)*ones(size(dayt,1),size(dayt,2)))
% plot(dayt,MODEL + sqrt(Q).' + sqrt(R1)*ones(size(dayt,1),size(dayt,2)),'g')
% title('MODEL AND OBSERVATIONS AND STANDARD DEVIATION OF NOISE')
% legend('MODEL', 'OBSERVATIONS','NOISE STANDARD DEVIATION')
% hold off


% n = 1;
% t = TIME(n);
% xs = To+Ta*cos(pi/w*(t-tm));
% x_(n) = xs;
% for n = 1:length(TIME)-1
%     t = TIME(n);
%     if t<=(ts-15/60)
%         if t<tm  %sin((t-tm)*pi/w)<0
%             x_(n+1) = To+(xs - To)*cos(pi/(w*4)) + sqrt(Ta^2 - (xs-To)^2) *sin(pi/(w*4));
%         else
%             x_(n+1) = To+(xs - To)*cos(pi/(w*4)) - sqrt(Ta^2 - (xs-To)^2) *sin(pi/(w*4));
%         end
%     elseif (t> (ts-15/60)) & (t<ts)
%         %x_(n+1) = (To + dT) + (Ta * cos(pi/w*(ts - tm)) - dT) *exp(-(t + 15/60 -ts)/k);
%         x_(n+1) = (To + dT) + ((xs-To)*cos(pi/w*(ts-t)) -sqrt(Ta^2 - (xs-To)^2)*sin(pi/w*(ts-t))- dT) *exp(-(t + 15/60 -ts)/k);
%     else
%         x_(n+1) = (To+dT)+(xs-To-dT)*exp(-1/4/k);
%     end
%     xs = x_(n+1);
% end
%
%
% figure
% plot(dayt,MODEL)
% hold on
% plot(dayt,x_,'r')
% title('MODEL AND MODEL recursive')
% legend('MODEL', 'MODEL recursive')
% hold off
%
% figure
% plot(dayt,MODEL-x_,'g')
% title('DIFFERENCE BETWEEN MODEL AND RECURSIVE MODEL')
% grid






% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % ct = TIME(1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % x(1) = To+Ta*cos(pi/w*(ct-tm)) %Initial state%+n(0)=0;n(1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % for nk=2:N,
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %n(nk-1) = 0;%#####################
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     ct = ct+15/60;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     if nk==2
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         pns = 0;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     else
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         pns = n(nk-2);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     if ct<ts,
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         if sin((ct-tm)*pi/(4*w))>0
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             x(nk) = To+(x(nk-1) - To-pns)*cos(pi/(w*4)) - sqrt(abs(Ta^2 - (x(nk-1)-To-pns)^2)) *sin(pi/(w*4)) + n(nk-1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         else
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             x(nk) = To+(x(nk-1) - To-pns)*cos(pi/(w*4)) + sqrt(abs(Ta^2 - (x(nk-1)-To-pns)^2)) *sin(pi/(w*4)) + n(nk-1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     else
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         x(nk) = (To+dT)+(x(nk-1)-To-dT-pns)*exp(-1/4/k)+n(nk-1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % end;


% if ct<=(ts-15/60)
%             if ct<tm  %sin((t-tm)*pi/w)<0
%                 x_(:,t+1) = real(To+(xf(:,t) - To)*cos(pi/(w*4)) + sqrt(Ta^2 - (xf(:,t)-To).^2) *sin(pi/(w*4))) + n(:,t+1);
%             else
%                 x_(:,t+1) = real(To+(xf(:,t) - To)*cos(pi/(w*4)) - sqrt(Ta^2 - (xf(:,t)-To).^2) *sin(pi/(w*4))) + n(:,t+1);
%             end
%         elseif (t> (ts-15/60)) & (t<ts)
%             %x_(:,t+1) = (To + dT) + (Ta * cos(pi/w*(ts - tm)) - dT) *exp(-(t + 15/60 -ts)/k);
%             x_(:,t+1) = real((To + dT) + ((xf(:,t)-To)*cos(pi/w*(ts-t)) -sqrt(Ta^2 - (xf(:,t)-To).^2)*sin(pi/w*(ts-t))- dT) *exp(-(t + 15/60 -ts)/k))  + n(:,t+1);
%         else
%             x_(:,t+1) = (To+dT)+(xf(:,t)-To-dT)*exp(-1/4/k)  + n(:,t+1);
% end

dC = cos(pi./w2.*(t-tm + 15/60)) - cos(pi./w1.*(t-tm));

% keep_t_ts1 = (t<=ts-15/60)*2;
% keep_t_ts2 = (t>(ts-15/60)) & (t<ts);
% keep_t_ts = keep_t_ts1 + keep_t_ts2; %0:Last condition, 1:Middle condition, 2:First condition

keep_t_ts1 = (t <= tm)*4;
keep_t_ts3 = ((t > tm) & (t <= (ts-15/60)))*2;
keep_t_ts = keep_t_ts1 + keep_t_ts3; %0:Last condition, 1:Middle condition, 2:First condition


%     [I_ts Jts] = (find(keep_t_ts1==2));


%     if t<=ts-15/60 %t<ts

% keep_t_tm = sign((t<tm)-0.5);


for j = 1:Ns

    Pr = zeros(size(t,1),size(t,2));
    keep_xf(1:size(t,1),1:size(t,2)) = xf(:,:,j);
    keep_process_noise(1:size(t,1),1:size(t,2)) = process_noise(:,:,j);
    %     if t<=(ts-15/60)  %Predict for the next sample
    %         if t<tm  %sin((t-tm)*pi/w)<0
    %Pr(find(keep_t_ts==2)) = real(To(find(keep_t_ts==2))+(xs(find(keep_t_ts==2)) - To(find(keep_t_ts==2))).*cos(pi./(w(find(keep_t_ts==2))*4)) + keep_t_tm(find(keep_t_ts==2)).*sqrt(Ta(find(keep_t_ts==2)).^2 - (xs(find(keep_t_ts==2))-To(find(keep_t_ts==2))).^2) .*sin(pi./(w(find(keep_t_ts==2))*4)));
    %Pr(keep_t_ts==2) = real(To(keep_t_ts==2)+(keep_xf(keep_t_ts==2) - To(keep_t_ts==2)).*cos(pi./(w(keep_t_ts==2)*4)) + keep_t_tm(keep_t_ts==2).*sqrt(Ta(keep_t_ts==2).^2 - (keep_xf(keep_t_ts==2)-To(keep_t_ts==2)).^2) .*sin(pi./(w(keep_t_ts==2)*4))) + keep_process_noise(keep_t_ts==2);
    Pr(keep_t_ts==4) = real(To(keep_t_ts==4)+(keep_xf(keep_t_ts==4) - To(keep_t_ts==4)).*cos(pi./(w1(keep_t_ts==4)*4)) + sqrt(Ta(keep_t_ts==4).^2 - (keep_xf(keep_t_ts==4)-To(keep_t_ts==4)).^2) .*sin(pi./(w1(keep_t_ts==4)*4)))  + keep_process_noise(keep_t_ts==4);

    Pr(keep_t_ts==3) = real(keep_xf(keep_t_ts==3) + Ta(keep_t_ts==3).*dC(keep_t_ts==3));



    Pr(keep_t_ts==2) = real(To(keep_t_ts==2)+(keep_xf(keep_t_ts==2) - To(keep_t_ts==2)).*cos(pi./(w2(keep_t_ts==2)*4)) - sqrt(Ta(keep_t_ts==2).^2 - (keep_xf(keep_t_ts==2)-To(keep_t_ts==2)).^2) .*sin(pi./(w2(keep_t_ts==2)*4)))  + keep_process_noise(keep_t_ts==2);


    Pr(keep_t_ts==1) = real(To(keep_t_ts==1) + ((keep_xf(keep_t_ts==1)-To(keep_t_ts==1)).*cos(pi./w2(keep_t_ts==1).*(ts(keep_t_ts==1)-t(keep_t_ts==1))) -sqrt(Ta(keep_t_ts==1).^2 - (keep_xf(keep_t_ts==1)-To(keep_t_ts==1)).^2).*sin(pi./w2(keep_t_ts==1).*(ts(keep_t_ts==1)-t(keep_t_ts==1)))) .*exp(-(t(keep_t_ts==1) + 15/60 -ts(keep_t_ts==1))./k(keep_t_ts==1)));


    %x_(:,:,timestamp+1) = Pr;
    %         else
    %             x_(:,:,n+1) = real(To+(xs - To).*cos(pi./(w*4)) - sqrt(Ta.^2 - (xs-To).^2) .*sin(pi./(w*4)));
    %         end
    %     elseif (t> (ts-15/60)) & (t<ts)
    %Pr(find(keep_t_ts==1)) = real((To(find(keep_t_ts==1)) + dT(find(keep_t_ts==1))) + ((xs(find(keep_t_ts==1))-To(find(keep_t_ts==1))).*cos(pi./w(find(keep_t_ts==1)).*(ts(find(keep_t_ts==1))-t(find(keep_t_ts==1)))) -sqrt(Ta(find(keep_t_ts==1)).^2 - (xs(find(keep_t_ts==1))-To(find(keep_t_ts==1))).^2).*sin(pi./w(find(keep_t_ts==1)).*(ts(find(keep_t_ts==1))-t(find(keep_t_ts==1))))- dT(find(keep_t_ts==1))) .*exp(-(t(find(keep_t_ts==1)) + 15/60 -ts(find(keep_t_ts==1)))./k(find(keep_t_ts==1))));
    %Pr(keep_t_ts==1) = real((To(keep_t_ts==1) + dT(keep_t_ts==1)) + ((keep_xf(keep_t_ts==1)-To(keep_t_ts==1)).*cos(pi./w(keep_t_ts==1).*(ts(keep_t_ts==1)-t(keep_t_ts==1))) -sqrt(Ta(keep_t_ts==1).^2 - (keep_xf(keep_t_ts==1)-To(keep_t_ts==1)).^2).*sin(pi./w(keep_t_ts==1).*(ts(keep_t_ts==1)-t(keep_t_ts==1)))- dT(keep_t_ts==1)) .*exp(-(t(keep_t_ts==1) + 15/60 -ts(keep_t_ts==1))./k(keep_t_ts==1))) + keep_process_noise(keep_t_ts==1);
    %     else
    %Pr(find(keep_t_ts==0)) = (To(find(keep_t_ts==0))+dT(find(keep_t_ts==0)))+(xs(find(keep_t_ts==0))-To(find(keep_t_ts==0))-dT(find(keep_t_ts==0))).*exp(-1/4./k(find(keep_t_ts==0)));
    %Pr(keep_t_ts==0) = (To(keep_t_ts==0)+dT(keep_t_ts==0))+(keep_xf(keep_t_ts==0)-To(keep_t_ts==0)-dT(keep_t_ts==0)).*exp(-1/4./k(keep_t_ts==0))+keep_process_noise(keep_t_ts==0);
    Pr(keep_t_ts==0) =  To(keep_t_ts==0)+ (keep_xf(keep_t_ts==0)-To(keep_t_ts==0)).*exp(-1/4./k(keep_t_ts==0)) +keep_process_noise(keep_t_ts==0);


    %     end

    x_(1:size(t,1),1:size(t,2),j) = Pr;

end
clear Pr keep_t_ts1 keep_t_ts2 keep_t_ts keep_t_tm keep_xf keep_process_noise process_noise

% clear n x_ t

% if cycle_out_scope==0


%     PARTICLECHECK = [];
%     for Ns=50:50:1000;        % Number of Monte Carlo samples per time step. % Number of samples; Number of particles
%     xf = [];
%     n = [];
%     x_ = [];
%
%     z = v4;  %Observation
%
%
%     R = R1; %Measurement noise covariance
%
%     Ns = 450;%800;

%v = sqrt(R)*randn(LENGTH_CURRENT_CYCLE,1); %The noise measurement covariance constant over the whole range of the cycle
%     n = sqrt(Q);%bs;%sqrt(Q)*randn(N,1);

%     Q1 = Q; %Process noise covariance



% ct = 6;
%xs = To+Ta*cos(pi/w*(t-tm));
% measnoise = 0.1;
% R1 = measnoise^2; % measurement error covariance
% R = R1;






% SAMPLE FROM THE PRIOR:
% =====================



%     clear n;
%P = 1;
%ct = 6;
%xf(:,1) = sqrt(P)*rand(Ns,1) + To;







%axis([6 30-15/60 min(z) max(z)])

%hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % checkTc = [Tc];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % checkinn = [inn(1)];
%figure
%     for t=1:LENGTH_CURRENT_CYCLE-1,
%         ct = TIME(t); %CURRENT TIME TO PREDICT FOR THE NEXT TIME%ct+15/60; %Start from index 2 because index 1 has been done before this loop
%         %xu(:,t) = predictstates(x(:,t),t,Q);
%
%
%         n(:,t+1) = sqrt(Q(t+1))*randn(size(xf(:,t)));
%         %n(:,t+1) = normrnd(0,sqrt(Q(t+1)),size(xf(:,t)));
%
%         R = R1; %So to get predicted residual covariance
%n(ixp(1),t+1) = sqrt(Q(t+1))*ones(size(xf(ixp(1),t)));%*randn(size(xf(:,t)));

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     if t==1
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         pns = 0;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     else
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         pns = n(:,t-1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %n(:,t) = 0;%#####################
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %n(:,t+1) = 0;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     if ct<ts,
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         if sin((ct-tm)*pi/(4*w))>0
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             x_(:,t+1) = To+(xf(:,t) - To - pns).*cos(pi/(w*4)) - sqrt(abs(Ta^2 - (xf(:,t)-To-pns).^2)) .*sin(pi/(w*4))+n(:,t);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         else
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             x_(:,t+1) = To+(xf(:,t) - To - pns).*cos(pi/(w*4)) + sqrt(abs(Ta^2 - (xf(:,t)-To-pns).^2)) .*sin(pi/(w*4))+n(:,t);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     else
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         x_(:,t+1) = (To+dT)+(xf(:,t)-To-dT - pns)*exp(-1/4/k)+n(:,t);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     end

%xf: filtered estimate
%x_:predicted estimate


%         if ct<=(ts-15/60)
%             if ct<tm  %sin((t-tm)*pi/w)<0
%                 x_(:,t+1) = real(To+(xf(:,t) - To)*cos(pi/(w*4)) + sqrt(Ta^2 - (xf(:,t)-To).^2) *sin(pi/(w*4))) + n(:,t+1);
%             else
%                 x_(:,t+1) = real(To+(xf(:,t) - To)*cos(pi/(w*4)) - sqrt(Ta^2 - (xf(:,t)-To).^2) *sin(pi/(w*4))) + n(:,t+1);
%             end
%         elseif (t> (ts-15/60)) & (t<ts)
%             %x_(:,t+1) = (To + dT) + (Ta * cos(pi/w*(ts - tm)) - dT) *exp(-(t + 15/60 -ts)/k);
%             x_(:,t+1) = real((To + dT) + ((xf(:,t)-To)*cos(pi/w*(ts-t)) -sqrt(Ta^2 - (xf(:,t)-To).^2)*sin(pi/w*(ts-t))- dT) *exp(-(t + 15/60 -ts)/k))  + n(:,t+1);
%         else
%             x_(:,t+1) = (To+dT)+(xf(:,t)-To-dT)*exp(-1/4/k)  + n(:,t+1);
%         end



%         m = x_(:,t+1);
%
%         prediction(t+1) = mean(x_(:,t+1)); %prediction estimate for standard SIR (equal weight, so as to use mean)
%         %prediction(t+1) = x_(ixp(1),t+1);
%         %     bins = 20;
%         %     [p,pos]=hist(x_(:,t+1),bins);
%         %     map=find(p==max(p));
%         %     prediction(t+1)=pos(map(1));
%         % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%         % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %FIRE DETECTION FOR PAPER 2.
%         inn(t+1) = z(t+1) - prediction(t+1);
%         S(t+1) = mean((m - prediction(t+1)).^2) + R; %Residual covariance: for standard SIR (equal weight, so as to use mean)


%         SSSPREDICTED = S(t+1)


%         sigma = sqrt(R);
%         its = find(x_(:,t+1)<=z(t+1));
%         y = exp(-(z(t+1)-x_(its,t+1)).^2/(2*sigma^2));
%         ur(t+1) = sum(y)


%         sigma = range(x_(:,t+1))/Ns
%         sigma = sqrt(R);
%         for s=1:Ns
%             itg(s) = quad(@(theta) gk(theta,x_(s,t+1),sigma),-Inf,z(t+1));
%         end
%         ur(t+1) = sum(itg)/Ns;


%         %sigma = range(x_(:,t+1))/Ns;
%         sigma = sqrt(R);%sqrt(S(t+1));
%
%         for s=1:Ns
%             %itg(s) = 1/2*(1 + erf((z(t+1) - x_(s,t+1))/sqrt(2*sigma^2)));
%             itg(s) = 1 - 1/2*erfc((z(t+1) - x_(s,t+1))/sqrt(2*sigma^2));
%         end
%         ur(t+1) = sum(itg)/Ns;
%
%         vr(t+1) = norminv(ur(t+1),0,1);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     K = 3;%20;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     prev = 1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     if t<2
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         D = (MNs(1:t+1,CYCLE) + MNs(1:t+1,CYCLE-prev))/2;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         Mn = mean(D);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         Sn = std(D);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         Tc = Mn + K*Sn;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %Tc = K;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %Tc = 5;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     else
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %D = b2(t-2:t) - b1(t-2:t);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         D = (MNs(t-1:t+1,CYCLE) + MNs(t-1:t+1,CYCLE-prev))/2;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         Mn = mean(D);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         Sn = std(D);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %Tc = min(Mn + K*Sn,2);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         Tc = Mn + K*Sn;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %Tc = mean(z());
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %Tc = 5;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     Tckeep(t) = Tc;


%         Tc = sqrt(S(t+1))*0.5;%100;

%         Tc = NUMBER_STANDARD_DEVIATION;
%
%         %R = R1;
%         if (vr(t+1)>Tc)%(inn(t+1)>Tc)
%
%             fire_detect(t+1) = 1;
%             % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             %R2 = R1;
%             % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             R2 = inn(t+1)^2;
%             %             R = inn(t+1)^2;
%             % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             %prediction(t+1) = x_(ixp(1),t+1);
%             % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             %R2 = cov(z(t+1)-x_(:,t+1));
%             % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             checkTc = [checkTc Tc];
%             % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%             % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             checkinn = [checkinn inn(t+1)];
%             %             R = 1/12*(335 - prediction(t+1) - Tc)^2; %Assuming a uniform distribution from Tc to (335 - prediction)
%             %Change only likelihood = change only R (increase this variance or make R = Inf (R=Inf, resampling is not important) so that the prior can have a priority)
%             %residual covariance can be changed inside here, but to keep it as predicted it is not implemented inside here
%
%             R = Inf;
%
%         else
%             fire_detect(t+1) = 0;
%             % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             R2 = R1; %MUST ALSO CHANGE ABOVE IF THERE IS  A CHANGE HERE, ON THE MEASUREMENT VARIANCE
%             % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             %R2 = inn(t+1)^2;
%             %Change only likelihood = change only R
%             %residual covariance can be changed inside here, but to keep it as predicted it is not implemented inside here
%             R = R1;
%         end

%         S1 = mean((m - prediction(t+1)).^2) + R; % Updated residual covariance %Residual covariance: for standard SIR (equal weight, so as to use mean)

%R = R1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         R = R2;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             %R2 = R1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%m = x_(:,t+1);%(x_(:,t).^(2))./20;

%m = (m - mean(m))/std(m) *(std(m)+sqrt(R)-sqrt(R1)) + mean(m);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%m = (m - mean(m))/std(m) *(std(m)*sqrt(R1/R2)) + mean(m);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     x_(:,t+1) = m;

%         %Calculate importance weights and normalize the weights
%         for s=1:Ns,
%             alpha = z(t+1).*ones(size(m))-m;
%             %q(s,t+1) = exp(-0.5*R^(-1)*(z(t+1)- m(s,1))^(2))./sum(exp(-0.5*R^(-1)*alpha.^(2)));
%             %sum(exp(-0.5*R^(-1)*alpha.^(2)))
%             q(s,t+1) = exp(-0.5*R^(-1)*(z(t+1)- m(s,1))^(2))/sum(exp(-0.5*R^(-1)*alpha.^(2)));
%         end;


%         prediction_Likelihood(t+1) = sum(q(:,t+1).*x_(:,t+1));
%         inn_Likelihood(t+1) = z(t+1) - prediction_Likelihood(t+1);
%         S_Likelihood(t+1) = mean((m - prediction_Likelihood(t+1)).^2) + R; %Residual covariance: for standard SIR (equal weight, so as to use mean)
%
%         SSS_REAL =  S_Likelihood(t+1)

%LIKELIHOOD ESTIMATE
%LPRED = [LPRED sum(alpha)];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     LPRED = [LPRED z(t+1)-prediction(t+1)];

%Update - Resampling algorithm



%     clf
%     plot(x_(:,t+1),q(:,t+1),'*')




%         %Systematic resampling
%         cdf_q = cumsum(q(:,t+1));
%         i = 1;
%         u(i) = rand*1/Ns;
%         for j = 1: Ns
%             u(j) = u(1) + 1/Ns*(j - 1);
%             while u(j) > cdf_q(i)%&(i<500)
%                 i = i + 1;
%             end
%             xf(j,t+1) = x_(i,t+1);
%             q(j,t+1) = 1/Ns;
%             %Assign parent keepi(j,t+1) = i;
%         end
%
%
%         %     u = rand(Ns+1,1);
%         %     r = -log(u);
%         %     cdf_u = cumsum(r);
%         %     i = 1;
%         %     j = 1;
%         %
%         %     while j <= Ns,
%         %         if (cdf_q(j)*cdf_u(Ns)) > cdf_u(i)
%         %             xf(i,t+1) = x_(j,t);
%         %             i = i+1;
%         %         else
%         %             j = j+1;
%         %         end;
%         %     end;
%
%         %%%%%
%         %MEAN estimate for the filtered estimate
%         estimation(t+1) = mean(xf(:,t+1));
%         xes = estimation(t+1);

%MAP (PEAK estimate for the filtered estimate)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             bins = 20;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             [p,pos]=hist(xf(:,t+1),bins);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             map=find(p==max(p));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             estimation(t+1)=pos(map(1));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             xes = estimation(t+1);





%xp = xf(:,t+1);
%ixp = find(xp==estimation(t+1));






%%%ANOTHER METHODS (CHECK ALSO ABOVE)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     if ct<ts,
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         if sin((ct-tm)*pi/w)>0
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             prediction(t+2) = To+(xes - To)*cos(pi/(w*4)) - sqrt((Ta^2 - (xes-To)^2)) *sin(pi/(w*4));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         else
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             prediction(t+2) = To+(xes - To)*cos(pi/(w*4)) + sqrt((Ta^2 - (xes-To)^2)) *sin(pi/(w*4));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     else
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         prediction(t+2) = (To+dT)+(xes-To-dT)*exp(-1/4/k);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     keepct(t) = ct;






% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     CHECKPRED(t,:) = [xes prediction(t+2)]

%     ixp = [];
%     xp = abs(estimation(t+1) - xf(:,t+1));
%     [yp,ixp] = min(xp);


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     LEST = [LEST z(t+1)-estimation(t+1)];


%     CHECKESTIMATION = [CHECKESTIMATION xf(ixp(1),t+1)]
%%%%%%%%%%%%%%prediction(t+1)=mean(xf(:,t+1));

%plot(t_(t+1),prediction(t+1 ),'*');

%     Inn(t+1) = z(t+1) - prediction(t+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     K = 1;
%     if t~=1
%
% %          if t<3
% %              D = bA(1,CYCLE) - bA(1,CYCLE-1);
% %              Mn = mean(D);
% %              Sn = std(D);
% %              Tc = Mn + K*Sn;
% %              %Tc = K;
% %              Tc
% %          else
% %              %D = b2(t-2:t) - b1(t-2:t);
% %              D = bA(t-1:t+1,CYCLE) - bA(t-1:t+1,CYCLE-1);
% %              Mn = mean(D);
% %              Sn = std(D);
% %              %Tc = min(Mn + K*Sn,2);
% %              Tc = Mn + K*Sn;
% %              %Tc = mean(z());
% %              Tc
% %              Q(t+1)
% %              Mn
% %              Sn
% %          end
%
%          if t<3
%              D = (MNs(1,CYCLE) + MNs(1,CYCLE-1))/2;
%              Mn = mean(D);
%              Sn = std(D);
%              Tc = Mn + K*Sn;
%              %Tc = K;
%              %Tc = 5;
%          else
%              %D = b2(t-2:t) - b1(t-2:t);
%              D = (MNs(t-1:t+1,CYCLE) + MNs(t-1:t+1,CYCLE-1))/2;
%              Mn = mean(D);
%              Sn = std(D);
%              %Tc = min(Mn + K*Sn,2);
%              Tc = Mn + K*Sn;
%              %Tc = mean(z());
%              %Tc = 5;
%
%          end
%
%
%
%
%          [t+1 Inn(t+1) Tc];
%
%
%
%          if (Inn(t+1)>Tc)
%              fire_detect(t+1) = 1;
%              Inn(t+1) = z(t)- prediction(:,t);
%              R = Inn(t+1)^2;
%              %Inn(t) = z(t+1)- prediction(:,t+1);
%              Inn(t+1) = z(t+1)- prediction(:,t+1);
%          else
%              fire_detect(t+1) = 0;
%              R = R1;
%          end
%     end
%     prediction(t+1) = z(t+1) - Inn(t+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% COMPUTE CENTROID, MAP AND VARIANCE ESTIMATES:
% ============================================

% Posterior mean estimate
%prediction = mean(xf);

%     % Posterior peak estimate
%     bins = 20;
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%xmap=zeros(N,1);
%     xmap = 0; %Added
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%for t=1:N
%     [p,pos]=hist(xf(:,t,1),bins);
%     map=find(p==max(p));
%     xmap(t,1)=pos(map(1));
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%end;

% Posterior standard deviation estimate
%xstd=std(xf);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%[mean((prediction-x).^2) mean((xmap.'-x).^2)]

% PLOT RESULTS:
% ============
%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%figure(2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%stairs(t_(1:t),prediction(1:t));
%stairs(t_,xmap);


%pause(1)


%     if t ~=1
%         %plot(t_(t),fire_detect(t)*10+280,'m-')
%         plot([t_(t-1) t_(t)], [fire_detect(t-1)*10+280 fire_detect(t)*10+280],'m-')
%     end
%stairs(1:length(x),z,'g-*',1:length(x),prediction,'r-+',1:length(x),xmap,'m-*')
%legend('True value','Posterior mean estimate','MAP estimate');

%%%%%%%%%%%%%%%%%%%%%%%%%%plot(t_(t),fire_detect*10+280,'m')


%ylabel('State estimate','fontsize',15)
%xlabel('Time','fontsize',15)
%axis([0 100 250 320]);
%hold off
%pause(1);

%         stResidual(t+1) = vr(t+1);%inn(t+1)/sqrt(S(t+1)); %Standardized residual using predicted residual covariance

%     end;


%         PARTICLECHECK = [PARTICLECHECK mean((estimation - z).^2)]; %For each cycle   PREDICTION_MSE = mean((prediction-v4).^2)
%         prediction_Likelihood
%         prediction
%         estimation

%     end


%     %NUMBER OF PARTICLES
%     figure
%     plot(50:50:1000,PARTICLECHECK)
%     title('ERROR AGAINST NUMBER OF PARTICLES')



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %OBSERVATION, PREDICTION AND STATE
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     hd= figure;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %startt = 4.50;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     t_ = TIME;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %plot(t_,estimation,'b')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     plot(t_,z,'r*');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     stairs(t_,prediction)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %plot(t_,prediction)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     xlabel('Time stamp');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     ylabel('Brightness temperature (Kelvin)');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%plot(t_,fire_detect*10+280,'m--')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %plot(t_,fire_detect*10+280,'m--')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     plot(t_,[fire_detect*10+mean(v4)],'m--')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     legend('Observed temperature','Predicted temperature','Condition state=(fire,no fire)')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     MSE_PREDICTION_SIR = mean(((z-prediction).*not(fire_detect)).^2)    %MSE considering samples not affected by FIRE
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % line([0 30],[280 280])
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % line([0 30],[290 290])
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % text(31,280,'no fire')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % text(31,290,'fire')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     line([0 30],[mean(v4) mean(v4)])
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     line([0 30],[mean(v4)+10 mean(v4)+10])
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     text(31,mean(v4),'no fire')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     text(31,mean(v4)+10,'fire')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %RESIDUAL AND RESIDUAL VARIANCE
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     plot(t_,inn,'m')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     title('RESIDUAL')% and its standard deviation')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     plot(t_,sqrt(S)*0.5,'c')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     plot(t_,-sqrt(S)*0.5,'c')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     plot(t_,z - prediction,'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     legend('RESIDUAL', 'STANDARD DEVIATION')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %ALTERNATIVE TEST STATISTICS COMPARED AGAINST STANDARDIZED RESIDUAL
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     plot(t_,vr,'b-')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     title('ALTENATIVE TEST STATISTIC')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     plot(t_,stResidual,'r-') %inn./sqrt(S)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %grid
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     legend('Alternative statistic','normalized residual')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % % % % % % % %     %TEST THE ALTERNATIVE TEST STATISTICS
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % % % % % % % %     figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % % % % % % % %     plot(t_,v,'b-')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % % % % % % % %     title('ALTERNATIVE TEST STATISTIC')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % % % % % % % %     hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % % % % % % % %     plot(t_,alpha./sqrt(S),'r-')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % % % % % % % %     hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %RESIDUAL DISTRIBUTION (HISTOGRAM)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     hist(inn)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     title('RESIDUAL DISTRIBUTION')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     sk = skewness(inn)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     kur = kurtosis(inn)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     [MUHAT SIGMAHAT] = normfit(stResidual) %Normal distribution
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     PARMHAT = evfit(stResidual) %Extreme value
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %STANDARDIZED RESIDUAL DISTRIBUTION (CDF)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     cdfplot(stResidual)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     [cdfF,valuex] = ecdf(stResidual);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     plot(valuex,cdfF,'r*')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     title('Standardized Residual CDF')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %NORMAL DISTRIBUTION REFERENCE CDF
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     mu = mean(stResidual);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     v = var(stResidual); %v is variance
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     referenceCDF = 1/2*(1+erf((stResidual-mu)/sqrt(2*v)));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %referenceCDF = normcdf(stResidual,mu,sqrt(v));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %KOLMOGOROV-SMIRNOV TEST
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %To choose distribution, find critical value or largest deviation from
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %the reference.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     HYPOTHESIS = kstest(stResidual,[stResidual.' referenceCDF.'])
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %HYPOTHESIS = lillietest(stResidual) %for only normal distribution
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     probplot(stResidual)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     title('ProbPlot')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     normplot(stResidual)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     title('NormPlot')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %       jbtest(stResidual) %Jarque-Bera test of normality
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %       chi2gof(stResidual) %Chi-square goodness-of-fit test (of normality)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %       lillietest(stResidual) %Lilliefors test of normality
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %       figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %       [FP,XIP]=ksdensity(stResidual);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %       plot(XIP,FP)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %       grid
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %       title('KSDENSITY')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %        %To check fitting with statistical toolbox dfittool
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %         save SAVESTDRESIDUAL stResidual
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     kstest      - Kolmogorov-Smirnov test for one sample.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     randn('state',sa);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % figure(2)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(prediction,z,'+')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % ylabel('True state','fontsize',15)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % xlabel('Posterior mean estimate','fontsize',15)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % c=-600:1:600;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(c,c,'r');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % axis([250 320 250 320]);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % figure(3)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(xmap,z,'+')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % ylabel('True state','fontsize',15)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % xlabel('MAP estimate','fontsize',15)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % c=-600:1:600;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(c,c,'r');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % axis([250 320 250 320]);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % stairs(t_,real(x_(1:96)))
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % grid
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(t_,y,'r*')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % xlabel('Time stamp');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % ylabel('temperature (Kelvin)');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(t_,fire_detect*10+280,'m')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % plot(t_,real(Inn),'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % grid
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % ff = find(fire_detect==1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % NUMBER_OF_FIRE = length(find(fire_detect==1))
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % disp('Fire at the following times:')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % t_(ff)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % subplot(211)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(t_,LPRED)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % %plot(t_,abs(fft(LPRED)))
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % grid
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(t_(20)*ones(1,10),0:9,'g-')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % %hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % subplot(212)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(t_,LEST,'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % %plot(t_,abs(fft(LEST)),'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(t_(20)*ones(1,10),0:9,'g-')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % title('Residuals')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % %hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %grid
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % randn('state',sa);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % size(Q)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % hist(LEST)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % grid
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(t_,LEST,'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % grid
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % for i=1:93
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     vs(i) = var(LEST((i:i+3)));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(vs,'g')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % grid
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % RRR = R
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % QQQ = sqrt(Q)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %FIRE DETECTION USING KALMAN FILTER ON FILTERED STATE RESIDUAL
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % Pd_ = 1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % Rd = 1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % xd_(1) = 0;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % Qd = 1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % fire(1) = 0;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % for i=1:95
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     Gd  = Pd_*inv(Pd_+Rd);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     alphad(i) = LEST(i) - xd_(i);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     xd_(i+1) = xd_(i) + Gd*alphad(i);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     if xd_(i+1)>sqrt(Rd)/2
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %         fire(i) = 1; Use fire_detect instead
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     else
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %         fire(i) = 0;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     Pd = (1-Gd)*Pd_;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     Pd_ = Pd + Qd;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(xd_)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % grid
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % mean(LEST)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % var(LEST)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % mean(alpha)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % var(alpha)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % figure(hd)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % %plot(t_(1:end-1),fire_detect*30 + 280,'g')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(t_,fire_detect*20 + 280,'r-')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % Tckeep
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % CHECKCHECK = [checkinn.' checkTc.']
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % size(z)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % size(prediction)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % mean((prediction(1:96) - z.').^2)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % RRRR = R1
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % plot(prediction(1:96),estimation,'k*')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % grid
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % MMMMM = mean((prediction(1:96) - z(1:96).').^2)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % PEPE = [prediction(1:96).' estimation.' z]
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % size(n)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % size(n(:,2))
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % CHECKPRED
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % keepct
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     % tsss = ts
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %REPORT FIRE EVENTS
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     ff = find(fire_detect==1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     NUMBER_OF_FIRE = length(ff)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     disp('Fire at the following times:')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     if NUMBER_OF_FIRE~=0
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         tffraction = t_(ff)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %         rightTime = [12 27 42 57];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %         fractionTime = [0 25 50 75];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %Change to hour and minute xx(xx) First 2 digits are for hours and
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         fractionT = (tffraction - floor(tffraction))*60;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %         for rT = 1:length(tffraction)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %             time_(rT) = floor(fractionT) + floor(tffraction(rT))*100;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         %         end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         for rT = 1:length(tffraction)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             %time_(rT,:) = sprintf('%0.2d:%0.2d',floor(tffraction(rT)),floor(fractionT)); %IN HOUR:MINUTE_
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             %time_(rT) = tffraction; %IN HOUR
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     else
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         time_=[];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     ff = find(fire_detect==1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     NUMBER_OF_FIRE = length(find(fire_detect==1))
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     disp('Fire at the following times:')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     if NUMBER_OF_FIRE~=0
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %         tffraction = t_(ff);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %         rightTime = [12 27 42 57];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %         fractionTime = [0 25 50 75];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %         fractionT = (tffraction - floor(tffraction))*100;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %         for rT = 1:length(tffraction)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %             time_(rT) = rightTime(find(fractionTime == fractionT(rT))) + floor(tffraction(rT))*100;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %         end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     else
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %         time_=[];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %     end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %time_
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %close all
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %t_(ff)

% end