% Day3:2---
% 18 6: Day 4
% Day4: (3) 4,(5)
% Day5:(1)
% Day4:(3) or (5)

close all
%Check for censoring in case of changing the degree of freedom for a region
clear all
clc
addpath('C:\Users\JOSINE\MODEL_CODE_SMALL'); addpath('..\ResultsChapter4')
EXAMPLEPIXEL = 29; %29;


KeepModelled = []; %Is commented but it can be uncommented

%clear all
%addpath('NDeconvolution');
addpath('..\UJ_PREDICTION1\deconvolution');
addpath('..\UJ_PREDICTION1\TimePrediction');
%addpath ..\DTC_ChangeDetection\Distributions
addpath('..\UJ_PREDICTION1\Distributions2');
addpath('..\UJ_PREDICTION1\Contextual');
addpath('..\UJ_PREDICTION1\Distributions2\Coefficients3');
addpath('..\UJ_PREDICTION1\Results');
%addpath('UJ_PREDICTION1\solarTimeLocation');
%addpath('models');
%addpath('MODEL');
%addpath('POTENTIAL');
load FNDATAREGIONFILE;


sa = randn('state');
sb = rand('state');

randn('state',0);
rand('state',0);

EXAMPLELOCATION = [
    208 65 10
    220 77 11
    210 76 10
    225 84 11
    224 75 11
    223 75 11
    221 71 11
    227 75 11
    226 76 11
    210 80 9
    219 77 9
    208 74 11
    206 80 10
    216 78 11
    216 81 11
    213 79 11
    209 73 11
    213 79 12
    209 69 11
    222 70 12
    219 78 11
    215 81 11
    215 82 12
    227 75 9
    219 73 11
    222 30 10
    186 121 9
    215 50 12
    218 75 12
    121 129 12
    121 130 12]; %(995,896) but also (996,896), (994,895) and (1000,894), all for cycle 12: case XTKalman does not detected a temperature saturation point.


CENTRE_ROW = EXAMPLELOCATION(EXAMPLEPIXEL,1)+778-1; %982; %1076; %1060; %1069; %1076; %1014; %1076; %1014;  %2749; %1014; %1076; %1076; %1018; %1030; %1050; %1035; %860; %1060; %930; %917; %1018;   %851; %1018; %851;  %894; %898; %904;  %898  894
CENTRE_COLUMN = EXAMPLELOCATION(EXAMPLEPIXEL,2)+822-1; %902; %886; %941; %870; %917; %941; %984; %941; %984; %942; %984; %941; %947;%886; %990; %884; %878; %1000; %930; %963; %979; %886; %966;  %886; %966;  %976; %980; %1074; %980  976
%RIGHT_CYCLE = 8:9 %4+8-1:5+8-1; %9:11; %11; %8; %11; %10;

START_CYCLE = 8;
RIGHT_CYCLE = 8:EXAMPLELOCATION(EXAMPLEPIXEL,3) %4+8-1:5+8-1; %9:11; %11; %8; %11; %10;
%Day1 = RIGHT_CYCLE = 9;

START_COLUMN = CENTRE_COLUMN;
END_COLUMN = CENTRE_COLUMN;
START_ROW = CENTRE_ROW;
END_ROW = CENTRE_ROW;



%%RESTORE BACK
% %
%addpath('C:\Users\JOSINE\MODEL_CODE_SMALL\ResultsChapter4')
load startIndex
% ROW = 982:1007;
% COLUMN = 886:906;
KEEPT_STARTINDEX = startIndex;
startIndex = KEEPT_STARTINDEX(CENTRE_ROW - 982 + 1, CENTRE_COLUMN - 886 + 1);
%load ExampleSIR28 startIndex  %26,27,28
% % % % % startIndex = 1;

STARTINDX = startIndex
clear startIndex
% %
%%%


%[TESTINGCYCLE_DATA, CHECK_LENGTH] = modelParameters(CENTRE_ROW,CENTRE_COLUMN,RIGHT_CYCLE);
[TESTINGCYCLE_DATA, CHECK_LENGTH, NUMBER_ROW, NUMBER_COLUMN, INDEX_ROW, INDEX_COLUMN KEEP_TIME] = dtc4_v_fire_dynamic_v6(CENTRE_ROW,CENTRE_COLUMN,RIGHT_CYCLE,FNDATAREGION,START_COLUMN,END_COLUMN,START_ROW,END_ROW,START_CYCLE); %Pooled

%save TEMPTESTINGCYCLE_DATA TESTINGCYCLE_DATA
%clear FNDATAREGION
%ADDED FOR FIRE DETECTION
%%%
%%%

FILTER = 0; %0:Ensemble Kalman, 1:SIR, 2:weak constraint 4D-Var
DECONVOLUTION = 0;%0:Don't perform DECONVOLUTION, 1:perform DECONVOLUTION
RESIDUAL = 0;    %0:Prediction Residual, 1:OL, 2:gELL
% if RESIDUAL ==0
%     threshold = 2; %inf;
% elseif RESIDUAL ==1
%     threshold = 2; %log(2);
% elseif RESIDUAL ==2
%     threshold = 0.5; %For standard %0.2; For sgELL
% else
%     display('Specify the RESIDUAL(FEATURE) type')
% end

EnKF_POTENTIAL = [-0.2397 -2.5700 0.5241]; %[-0.4302 -1.1203 -0.2020];  %GOT01:[-0.3686 -1.4472 0.0314];
SIR_POTENTIAL = [-0.1701 -3.8721 0.8715]; %[-0.3341 -1.7016 0.1521];
w4DVar_POTENTIAL = [-0.3490 -2.6660  0.2155];
NUMBER_PARTICLE_SIR = 51; %450;
NUMBER_ENSEMBLE_MEMBERS = 51; %450;
ASSIMILATION_WINDOW_LENGTH = 5;

XTKALMAN_CONTEXTUAL = [-0.8610 -1.9330 0.3503];
SIR_CONTEXTUAL = [-0.7950 -2.1950 0.5336];


EXPONENT = -3.5; %[-6 downto -0.4343]
%for EXPONENT = -8:0.5:-1

if FILTER == 0
    Pfa = 10^EXPONENT;
    THRESHOLD_COEFFICIENT_POTENTIAL = EnKF_POTENTIAL(1)*log(Pfa) + EnKF_POTENTIAL(2)*Pfa + EnKF_POTENTIAL(3); %Gumbel, with Number of samples ignored as a variate.
    
    
    %Pfa = 10^(-6);
    PfaT = Pfa;
    Directional_Pfa = roots([1 -32/7 8 -32/5 2 0 0 0 -PfaT/35]);  %4 out of 8
    Pfac(1) = Directional_Pfa(find((Directional_Pfa - abs(real(Directional_Pfa)))==0));
    
    Directional_Pfa = roots([1 -5/2 5/3 0 0 -PfaT/6]); %3 out of 5
    Pfac(2) = Directional_Pfa(find((Directional_Pfa - abs(real(Directional_Pfa)))==0));
    
    Directional_Pfa = roots([1 -3/2 0 -PfaT/(-2)]); %2 out of 3
    Pfac(3) = Directional_Pfa(find((Directional_Pfa - abs(real(Directional_Pfa)))==0 & Directional_Pfa<=1));
    
    
    Pfac = Pfac./(1 + sqrt(1 - 2*Pfac));
    THRESHOLD_COEFFICIENT_CONTEXTUAL = XTKALMAN_CONTEXTUAL(1)*log(Pfac) + XTKALMAN_CONTEXTUAL(2)*Pfac + XTKALMAN_CONTEXTUAL(3); %Logistic, with Number of samples ignored as a variate.
    
    disp('Ensemble Kalman filter')
    
    
elseif FILTER == 1
    Pfa = 10^EXPONENT;
    THRESHOLD_COEFFICIENT_POTENTIAL = SIR_POTENTIAL(1)*log(Pfa) + SIR_POTENTIAL(2)*Pfa + SIR_POTENTIAL(3); %Gumbel, with Number of samples ignored as a variate.
    
    
    %Pfa = 10^(-6);
    PfaT = Pfa;
    Directional_Pfa = roots([1 -32/7 8 -32/5 2 0 0 0 -PfaT/35]);  %4 out of 8
    Pfac(1) = Directional_Pfa(find((Directional_Pfa - abs(real(Directional_Pfa)))==0));
    
    Directional_Pfa = roots([1 -5/2 5/3 0 0 -PfaT/6]); %3 out of 5
    Pfac(2) = Directional_Pfa(find((Directional_Pfa - abs(real(Directional_Pfa)))==0));
    
    Directional_Pfa = roots([1 -3/2 0 -PfaT/(-2)]); %2 out of 3
    Pfac(3) = Directional_Pfa(find((Directional_Pfa - abs(real(Directional_Pfa)))==0 & Directional_Pfa<=1));
    
    
    Pfac = Pfac./(1 + sqrt(1 - 2*Pfac));
    THRESHOLD_COEFFICIENT_CONTEXTUAL = SIR_CONTEXTUAL(1)*log(Pfac) + SIR_CONTEXTUAL(2)*Pfac + SIR_CONTEXTUAL(3); %Logistic, with Number of samples ignored as a variate.
    
    disp('SIR filter')
    
    
elseif FILTER == 2
    Pfa = 10^EXPONENT;
    THRESHOLD_COEFFICIENT_POTENTIAL = w4DVar_POTENTIAL(1)*log(Pfa) + w4DVar_POTENTIAL(2)*Pfa + w4DVar_POTENTIAL(3); %Gumbel, with Number of samples ignored as a variate.
    
    
    %Insert also the contextual
    
    disp('4DVar')
    
else
    disp('FILTER must be FILTER=0:EnKF and FILTER=1:SIR and FILTER=2:4DVar')
end


%%%
%%%

%TESTINGCYCLE_DATA.START_INDEX = 2; Only for pixel number 26 %STARTINDX; %2;
TESTINGCYCLE_DATA.START_INDEX = STARTINDX; %2;
%The starting indices for each pixels
n = zeros(NUMBER_ROW,NUMBER_COLUMN);
for jN = 1:NUMBER_ROW
    for iN = 1:NUMBER_COLUMN
        n(jN,iN) = TESTINGCYCLE_DATA(jN,iN).START_INDEX;
    end
end
startIndex = n;

% if cycle_out_scope==0

x_ = [];


%ADDED FOR FIRE DETECTION
%%
%%
prediction = [];
predictedObservation = [];
Kn_ = [];
EnsPX_ = []; %Ensemble perturbations
KeepAlpha = [];
KeepScaledResidual = [];
Keep_Q1 = [];
Keep_Inn = [];
Keep_Q2 = [];
Keep_S = [];
KeepFireDetect = [];
FLAG_BLURRED_PIXEL_NOTCONSIDERED = []; %Pixels which are not considered during entropy calculation
Bo = [];
Keepxs = [];
FEATURE1 = []; %Residual at launch time of a filter
y_predictedKeep = [];

INDEX_DEBLURRED_CYCLE = 0;
RESTOREDImage = [];
restored = [];

DPSF = [];
DEXIT = [];
DFIREDETECT = [];
DFIRETIMESTAMP = [];

NUMBER_TIMESTAMP = min(CHECK_LENGTH(:)) - max(n(:)) + 1;
%NUMBER_TIMESTAMP = min((length(TESTINGCYCLE_DATA(1,1).TEST_CYCLE_DATA) - n)+1); %300;
%NUMBER_TIMESTAMP = 300;
%for timestamp = 1:NUMBER_TIMESTAMP
SQ_LENGTH  = NUMBER_TIMESTAMP;
KeepJJ = [];

%%
%%


KEEP_MOD_SEQ = [];
%figure(10)
%hold on

%SQ_LENGTH = length(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA); %573
% if FILTER == 2
%     TESTINGCYCLE_DATA_LINEAR = DTCLinearModel(TESTINGCYCLE_DATA);
%     %clear TESTINGCYCLE_DATA---ENABLE AND COPY ROW AND COLUMN AS FILED OF
%     %TESTINGCYCLE_DATA_LINEAR
% end

if FILTER ==2
    %END_SIMULATION = -ASSIMILATION_WINDOW_LENGTH;%-2;%-60;%-80%-ASSIMILATION_WINDOW_LENGTH;
    START_SIMULATION = ASSIMILATION_WINDOW_LENGTH;%-2;%-60;%-80%-ASSIMILATION_WINDOW_LENGTH;
    n = n + ASSIMILATION_WINDOW_LENGTH
    KeepFireDetect(1:NUMBER_ROW,1:NUMBER_COLUMN,1:ASSIMILATION_WINDOW_LENGTH+1) = zeros(NUMBER_ROW,NUMBER_COLUMN,ASSIMILATION_WINDOW_LENGTH+1);
else
    START_SIMULATION = 0;
end


%for timestamp = 1:SQ_LENGTH +END_SIMULATION %-1%(-1 due to SIR)%length(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA)
%for timestamp = 1:95 +END_SIMULATION %-1%(-1 due to SIR)%length(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA)
%for timestamp = 1+START_SIMULATION:95  %-1%(-1 due to SIR)%length(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA)
for timestamp = 1+START_SIMULATION:SQ_LENGTH-1  %Analysis times %-1%(-1 due to SIR)%length(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA)
    %ADDED FOR FIRE DETECTION
    %%%
    %%%
    
    if FILTER == 0
        if timestamp>96 %Start estimating Location and Scale parameter of previous residual from timestamp=97
            
            %THRESHOLD_POTENTIAL = getThresholdPotential(setFireSampleToMissing(KeepScaledResidual,KeepFireDetect),timestamp,THRESHOLD_COEFFICIENT_POTENTIAL);
            
            %THRESHOLD_POTENTIAL = threshold;
            %THRESHOLD_POTENTIAL = getThresholdPotential2(setFireSampleToMissing(KeepScaledResidual,KeepFireDetect),timestamp,THRESHOLD_COEFFICIENT_POTENTIAL, LATEST_THRESHOLD_POTENTIAL);
            THRESHOLD_POTENTIAL = getThresholdPotential2(setFireSampleToMissing(PRkeep,KeepFireDetect),timestamp,THRESHOLD_COEFFICIENT_POTENTIAL, LATEST_THRESHOLD_POTENTIAL);
            LATEST_THRESHOLD_POTENTIAL = THRESHOLD_POTENTIAL;
        else
            %0.0831
            
            %THRESHOLD_POTENTIAL = threshold;
            THRESHOLD_POTENTIAL = sqrt(6*0.8833)/pi*THRESHOLD_COEFFICIENT_POTENTIAL*ones(size(TESTINGCYCLE_DATA,1),size(TESTINGCYCLE_DATA,2)); %Assumption of location 0 and variance 1
            LATEST_THRESHOLD_POTENTIAL = THRESHOLD_POTENTIAL;
        end
        %THRESHOLD_POTENTIAL = 0.75;
        
        %Analysis
        %[KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual y cycle_out_scope cycle_reported filtered Gf Keep_Q2 EnsPX_ prediction predictedObservation t] = stdEnkalmanDECONV10_v3(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,NUMBER_ENSEMBLE_MEMBERS, EnsPX_, prediction, predictedObservation);
        %[KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual y cycle_out_scope cycle_reported filtered Gf Keep_Q2 EnsPX_ prediction predictedObservation t] = stdEnkalmanDECONV10_v4(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,NUMBER_ENSEMBLE_MEMBERS, EnsPX_, prediction, predictedObservation);
        %[KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual y cycle_out_scope cycle_reported filtered Gf Keep_Q2 EnsPX_ prediction predictedObservation t FEATURE1 y_predictedKeep] = stdEnkalmanDECONV10_v5(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,NUMBER_ENSEMBLE_MEMBERS, EnsPX_, prediction, predictedObservation, threshold, FEATURE1, RESIDUAL, y_predictedKeep);
        %[KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual y cycle_out_scope cycle_reported filtered Gf Keep_Q2 EnsPX_ prediction predictedObservation t FEATURE1 y_predictedKeep] = stdEnkalmanDECONV10_v6(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,NUMBER_ENSEMBLE_MEMBERS, EnsPX_, prediction, predictedObservation, threshold, FEATURE1, RESIDUAL, y_predictedKeep); %Right Q
        %[KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual y cycle_out_scope cycle_reported filtered Gf Keep_Q2 EnsPX_ prediction predictedObservation t FEATURE1 y_predictedKeep] = stdEnkalmanDECONV10_v7(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,NUMBER_ENSEMBLE_MEMBERS, EnsPX_, prediction, predictedObservation, threshold, FEATURE1, RESIDUAL, y_predictedKeep); %Right R1c
        [KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual y cycle_out_scope cycle_reported filtered Gf Keep_Q2 EnsPX_ prediction predictedObservation t FEATURE1 y_predictedKeep] = stdEnkalmanDECONV10_v11(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,NUMBER_ENSEMBLE_MEMBERS, EnsPX_, prediction, predictedObservation, THRESHOLD_POTENTIAL, FEATURE1, RESIDUAL, y_predictedKeep); %Right R1c
        %========
        %========
        %========
        %         tk(:,:,timestamp)= t;
        %         %tr(:,:,timestamp+1) = tr(:,:,timestamp)+15/60;
        %         ykeep(:,:,timestamp) = y;
        %
        %
        %
        %         pd = 96*3; %pd: %number of timestamps %3 days = 96*3 samples to forecast
        %
        %         for forecastSteps = 1:pd
        %
        %             Ft = 1;
        %             xp = Ft^(forecastSteps-1)* filtered; %xp or xsmooth
        %
        %             %[NUMBER_ROW NUMBER_COLUMN] = size(TESTINGCYCLE_DATA);
        %             ynew = zeros(NUMBER_ROW,NUMBER_COLUMN);
        %             for jN = 1:NUMBER_ROW
        %                 for iN = 1:NUMBER_COLUMN
        %                     nnew = n + forecastSteps;
        %                     ynew(jN,iN) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_DATA(nnew(jN,iN));
        %                     tcurrent(jN,iN) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME(nnew(jN,iN));
        %                 end
        %             end
        %             ynewkeep(:,:,forecastSteps) = ynew;
        %             y_extended = extend(ynew,NUMBER_ENSEMBLE_MEMBERS);
        %
        %             %tcurrent = t+15/60*forecastSteps;
        %             tkcurrent(:,:,forecastSteps) = tcurrent; %+ 24 *(1+floor(forecastSteps/96));
        %             y_predicted(forecastSteps,:,:) = ensembleforecast(mean(xp,3),y_extended(1),tcurrent);
        %             %y_predicted [pd x Pixels x Ensemble size]
        %
        %         end
        %
        %
        %
        %         for le = 1:size(tkcurrent,3)
        %             tk(:,:,timestamp+le) = tkcurrent(:,:,le);
        %             ykeep(:,:,timestamp+le) = ynewkeep(:,:,le);
        %             %predictedObservation(:,:,timestamp+le)= y_predicted(:,:,le);
        %         end
        
        
        %========
        %========
        %========
        
        
        %Start detection using threshold at timestamp = 97, in order to get 96
        %samples to use for BLU estimate. Before timestamp = 97, use estimated parameters as
        %Location = 0 and Scale = 1
    elseif FILTER ==1
        if timestamp>96 %Start estimating Location and Scale parameter of previous residual from timestamp=97
            
            %THRESHOLD_POTENTIAL = getThresholdPotential(setFireSampleToMissing(KeepScaledResidual,KeepFireDetect),timestamp,THRESHOLD_COEFFICIENT_POTENTIAL);
            
            %THRESHOLD_POTENTIAL = threshold;
            %THRESHOLD_POTENTIAL = getThresholdPotential2(setFireSampleToMissing(KeepScaledResidual,KeepFireDetect),timestamp,THRESHOLD_COEFFICIENT_POTENTIAL, LATEST_THRESHOLD_POTENTIAL);
            THRESHOLD_POTENTIAL = getThresholdPotential2(setFireSampleToMissing(PRkeep,KeepFireDetect),timestamp,THRESHOLD_COEFFICIENT_POTENTIAL, LATEST_THRESHOLD_POTENTIAL);
            LATEST_THRESHOLD_POTENTIAL = THRESHOLD_POTENTIAL;
        else
            %0.0837
            
            %THRESHOLD_POTENTIAL = threshold;
            THRESHOLD_POTENTIAL = sqrt(6*1.9776)/pi*THRESHOLD_COEFFICIENT_POTENTIAL*ones(size(TESTINGCYCLE_DATA,1),size(TESTINGCYCLE_DATA,2));
            LATEST_THRESHOLD_POTENTIAL = THRESHOLD_POTENTIAL;
        end
        %THRESHOLD_POTENTIAL = 10;
        %THRESHOLD_COEFFICIENT_POTENTIAL
        %Get Y
        %             THRESHOLD_POTENTIAL = getThresholdPotential(KeepScaledResidual,timestamp,THRESHOLD_COEFFICIENT_POTENTIAL);
        %parameters = estimatedParameters(previousScaledResidual,0); %Use test33.m to get previousScaledResidual
        
        %Give this function the threshold cofficients and estimated parameters
        %Also to the function in decnvolution
        
        %[KeepFireDetect x_ prediction Keep_Q1 Keep_Inn Keep_S stResidual y cycle_out_scope cycle_reported estimation Keep_Q2 predictedObservation t filtered] = stdsirDECONV10_v3(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,prediction,Keep_Q1,Keep_Inn,Keep_S,KeepFireDetect,0,[],Keep_Q2, NUMBER_PARTICLE_SIR, predictedObservation);
        %[KeepFireDetect x_ prediction Keep_Q1 Keep_Inn Keep_S stResidual y cycle_out_scope cycle_reported estimation Keep_Q2 predictedObservation t filtered] = stdsirDECONV10_v4(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,prediction,Keep_Q1,Keep_Inn,Keep_S,KeepFireDetect,0,[],Keep_Q2, NUMBER_PARTICLE_SIR, predictedObservation);
        %[KeepFireDetect x_ prediction Keep_Q1 Keep_Inn Keep_S stResidual y cycle_out_scope cycle_reported estimation Keep_Q2 predictedObservation t filtered FEATURE1 y_predictedKeep] = stdsirDECONV10_v5(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,prediction,Keep_Q1,Keep_Inn,Keep_S,KeepFireDetect,0,[],Keep_Q2, NUMBER_PARTICLE_SIR, predictedObservation, threshold, FEATURE1, RESIDUAL, y_predictedKeep);
        %[KeepFireDetect x_ prediction Keep_Q1 Keep_Inn Keep_S stResidual y cycle_out_scope cycle_reported estimation Keep_Q2 predictedObservation t filtered FEATURE1 y_predictedKeep] = stdsirDECONV10_v6(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,prediction,Keep_Q1,Keep_Inn,Keep_S,KeepFireDetect,0,[],Keep_Q2, NUMBER_PARTICLE_SIR, predictedObservation, threshold, FEATURE1, RESIDUAL, y_predictedKeep); %Right Q
        %[KeepFireDetect x_ prediction Keep_Q1 Keep_Inn Keep_S stResidual y cycle_out_scope cycle_reported estimation Keep_Q2 predictedObservation t filtered FEATURE1 y_predictedKeep] = stdsirDECONV10_v7(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,prediction,Keep_Q1,Keep_Inn,Keep_S,KeepFireDetect,0,[],Keep_Q2, NUMBER_PARTICLE_SIR, predictedObservation, threshold, FEATURE1, RESIDUAL, y_predictedKeep);%Right R1c
        [KeepFireDetect x_ prediction Keep_Q1 Keep_Inn Keep_S stResidual y cycle_out_scope cycle_reported estimation Keep_Q2 predictedObservation t filtered FEATURE1 y_predictedKeep] = stdsirDECONV10_v11(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,prediction,Keep_Q1,Keep_Inn,Keep_S,KeepFireDetect,0,[],Keep_Q2, NUMBER_PARTICLE_SIR, predictedObservation, THRESHOLD_POTENTIAL, FEATURE1, RESIDUAL, y_predictedKeep);%Right R1c
        %filtered = estimation;
        
    elseif FILTER == 2;
        if timestamp>96 %Start estimating Location and Scale parameter of previous residual from timestamp=97
            %THRESHOLD_POTENTIAL = getThresholdPotential(setFireSampleToMissing(KeepScaledResidual,KeepFireDetect),timestamp,THRESHOLD_COEFFICIENT_POTENTIAL);
            
            %THRESHOLD_POTENTIAL = threshold;
            %THRESHOLD_POTENTIAL = getThresholdPotential2(setFireSampleToMissing(KeepScaledResidual,KeepFireDetect),timestamp,THRESHOLD_COEFFICIENT_POTENTIAL, LATEST_THRESHOLD_POTENTIAL);
            THRESHOLD_POTENTIAL = getThresholdPotential2(setFireSampleToMissing(PRkeep,KeepFireDetect),timestamp,THRESHOLD_COEFFICIENT_POTENTIAL, LATEST_THRESHOLD_POTENTIAL);
            LATEST_THRESHOLD_POTENTIAL = THRESHOLD_POTENTIAL;
        else
            %0.0504
            
            %THRESHOLD_POTENTIAL = threshold;
            THRESHOLD_POTENTIAL =  sqrt(6*0.7904)/pi*THRESHOLD_COEFFICIENT_POTENTIAL*ones(size(TESTINGCYCLE_DATA,1),size(TESTINGCYCLE_DATA,2)); %Assumption of location 0 and variance 1
            LATEST_THRESHOLD_POTENTIAL = THRESHOLD_POTENTIAL;
        end
        
        NUMBER_ENSEMBLE_MEMBERS = 51;
        %clear TESTINGCYCLE_DATA
        
        %THRESHOLD_POTENTIAL = 0.7354;
        
        
        %[KeepFireDetect prediction Kn_ KeepAlpha Keep_S stResidual y cycle_out_scope cycle_reported filtered Gf Keep_Q2] = stdw4DVarDECONV10_v3(TESTINGCYCLE_DATA_LINEAR,THRESHOLD_POTENTIAL,timestamp,n,prediction,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,ASSIMILATION_WINDOW_LENGTH);
        %[KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual z cycle_out_scope cycle_reported filtered Gf Keep_Q2 EnsPX_ prediction predictedObservation Bo] = stdw4DVarDECONV10_v3(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,ASSIMILATION_WINDOW_LENGTH, EnsPX_, prediction, predictedObservation, Bo);
        %[KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual z cycle_out_scope cycle_reported filtered Gf Keep_Q2 EnsPX_ prediction predictedObservation Bo KeepJJ t] = stdw4DVarDECONV10_v4(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,ASSIMILATION_WINDOW_LENGTH, EnsPX_, prediction, predictedObservation, Bo, KeepJJ);
        %[KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual z cycle_out_scope cycle_reported filtered Gf Keep_Q2 EnsPX_ prediction predictedObservation Bo KeepJJ t] = stdw4DVarDECONV10_v5(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,ASSIMILATION_WINDOW_LENGTH, EnsPX_, prediction, predictedObservation, Bo, KeepJJ);
        %[KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual z cycle_out_scope cycle_reported xs Gf Keep_Q2 EnsPX_ prediction predictedObservation Bo KeepJJ t Keepxs ht Rta] = stdw4DVarDECONV10_v6(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,ASSIMILATION_WINDOW_LENGTH, EnsPX_, prediction, predictedObservation, Bo, KeepJJ,Keepxs);
        %[KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual z cycle_out_scope cycle_reported xs Gf Keep_Q2 EnsPX_ prediction predictedObservation Bo KeepJJ t Keepxs ht Rta FEATURE1] = stdw4DVarDECONV10_v7(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,ASSIMILATION_WINDOW_LENGTH, EnsPX_, prediction, predictedObservation, Bo, KeepJJ,Keepxs, threshold, FEATURE1, RESIDUAL, NUMBER_ENSEMBLE_MEMBERS);
        %[KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual z cycle_out_scope cycle_reported xs Gf Keep_Q2 EnsPX_ prediction predictedObservation Bo KeepJJ t Keepxs ht Rta FEATURE1 y_predictedKeep] = stdw4DVarDECONV10_v8(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,ASSIMILATION_WINDOW_LENGTH, EnsPX_, prediction, predictedObservation, Bo, KeepJJ,Keepxs, threshold, FEATURE1, RESIDUAL, NUMBER_ENSEMBLE_MEMBERS, y_predictedKeep);
        %[KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual z cycle_out_scope cycle_reported xs Gf Keep_Q2 EnsPX_ prediction predictedObservation Bo KeepJJ t Keepxs ht Rta FEATURE1 y_predictedKeep] = stdw4DVarDECONV10_v9(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,ASSIMILATION_WINDOW_LENGTH, EnsPX_, prediction, predictedObservation, Bo, KeepJJ,Keepxs, threshold, FEATURE1, RESIDUAL, NUMBER_ENSEMBLE_MEMBERS, y_predictedKeep);
        %[KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual z cycle_out_scope cycle_reported xs Gf Keep_Q2 EnsPX_ prediction predictedObservation Bo KeepJJ t Keepxs ht Rta FEATURE1 y_predictedKeep] = stdw4DVarDECONV10_v10(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,ASSIMILATION_WINDOW_LENGTH, EnsPX_, prediction, predictedObservation, Bo, KeepJJ,Keepxs, threshold, FEATURE1, RESIDUAL, NUMBER_ENSEMBLE_MEMBERS, y_predictedKeep); %Right Q and R1c
        [KeepFireDetect x_ Kn_ KeepAlpha Keep_S stResidual z cycle_out_scope cycle_reported xs Gf Keep_Q2 EnsPX_ prediction predictedObservation Bo KeepJJ t Keepxs ht Rta FEATURE1 y_predictedKeep] = stdw4DVarDECONV10_v11(TESTINGCYCLE_DATA,THRESHOLD_POTENTIAL,timestamp,n,x_,Kn_,KeepAlpha,Keep_S,KeepFireDetect,0,[],Keep_Q2,ASSIMILATION_WINDOW_LENGTH, EnsPX_, prediction, predictedObservation, Bo, KeepJJ,Keepxs, THRESHOLD_POTENTIAL, FEATURE1, RESIDUAL, NUMBER_ENSEMBLE_MEMBERS, y_predictedKeep); %Right Q and R1c
        %y = z(:,:,1);
        y = z;
        
        
        %ESTIMATE(:,:,timestamp) = filtered;
        %OBSERVATION(:,:,timestamp) = y;
        
    else
        disp('FILTER=0:Extended Kalman, FILTER=1:SIR, FILTER=2:4DVar')
    end
    
    
    
    %=============
    %=============
    %=============
    
    %[x_  Kn_ KeepAlpha Keep_S stResidual filtered Gf EnsPX_ prediction] = stdEnkalmanDECONV10_v4(n,x_,Kn_,KeepAlpha,Keep_S,EnsPX_, prediction  );
    %[x_ prediction Keep_Q1 Keep_Inn Keep_S stResidual estimation] = stdsirDECONV10_v4(n,x_,prediction,Keep_Q1,Keep_Inn,Keep_S,,                     );
    
    if (FILTER==2) & (timestamp == ASSIMILATION_WINDOW_LENGTH + 1)
        tk = t;
        %tr(:,:,timestamp+1) = tr(:,:,timestamp)+15/60;
        ykeep = y;
        %Rt = squeeze(Keep_Q2(:,:,timestamp));
    else
        tk(:,:,timestamp) = t;
        ykeep(:,:,timestamp) = y;
    end
    
    
    if FILTER==2
        for pixelnumber = 1:size(x_,2)
            hto = ht(1:6,pixelnumber);
            htc = ht(7:12,pixelnumber);
            hta = [hto.';htc.'];
            CovObs = hta*Bo(1:6,1+(pixelnumber-1)*6:pixelnumber*6)*hta.' + diag(Rta(:,pixelnumber));
            Ao(1:6,1+(pixelnumber-1)*6:pixelnumber*6) = (eye(6,6) - Bo(1:6,1+(pixelnumber-1)*6:pixelnumber*6)*hta.'*inv(CovObs)*hta)*Bo(1:6,1+(pixelnumber-1)*6:pixelnumber*6);
        end
    end
    %pd = 96*3; %pd: %number of timestamps %3 days = 96*3 samples to forecast
    pd = 1; %pd: %number of timestamps %3 days = 96*3 samples to forecast
    
    
    for forecastSteps = 1:pd
        
        %Ft = 1;
        %xp = Ft^(forecastSteps-1)* filtered; %xp or xsmooth
        
        %[NUMBER_ROW NUMBER_COLUMN] = size(TESTINGCYCLE_DATA);
        ynew = zeros(NUMBER_ROW,NUMBER_COLUMN);
        for jN = 1:NUMBER_ROW
            for iN = 1:NUMBER_COLUMN
                nnew = n + forecastSteps;
                ynew(jN,iN) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_DATA(nnew(jN,iN));
                %tcurrent(jN,iN) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME(nnew(jN,iN));
                %tHorizonnew(jN,iN) = TESTINGCYCLE_DATA(jN,iN).tmorningHorizon(nnew(jN,iN));
                %observedThermalSunrise(jN,iN) = TESTINGCYCLE_DATA(jN,iN).tSunriseDay(nnew(jN,iN));
                
            end
        end
        ynewkeep(:,:,forecastSteps) = ynew;
        %tkcurrent(:,:,forecastSteps) = tcurrent;
        
        %tHorizonnewkeep(:,:,forecaSteps) = tHorizonnew;
        %observedThermalSunrisekeep(:,:,forecaSteps) = observedThermalSunrise;
        
        %         if timestamp == 93
        %             ynewkeep(:,:,forecastSteps) = ynew + 0;
        %         end
        
    end
    
    if FILTER==0
        %y_extended = extend(ynew,NUMBER_ENSEMBLE_MEMBERS);
        [tkcurrent y_predictedkeep SHIFT] = fixTime(TESTINGCYCLE_DATA,n,pd,filtered,y,NUMBER_ENSEMBLE_MEMBERS);
        %         mean(filtered,3)
        %         mean(y_predictedkeep,3)
    elseif FILTER==1
        %y_extended = extend(ynew,NUMBER_PARTICLE_SIR);
        [tkcurrent y_predictedkeep SHIFT] = fixTime(TESTINGCYCLE_DATA,n,pd,filtered,y,NUMBER_PARTICLE_SIR);
    else
        
        NUMBER_ENSEMBLE_MEMBERS = 51;
        
        
        
        if timestamp>25
            %timestamp
            filtered = createPerturbedAnalysis(Keepxs,NUMBER_ENSEMBLE_MEMBERS,Ao);
        else
            %disp('NOT YET')
            %Ao
            
            filtered = createPerturbedAnalysisMC(Keepxs,NUMBER_ENSEMBLE_MEMBERS,Ao);
        end
        %y_extended = extend(ynew,1);
        %display('SPECIFY FILTER')
        
        %CREATE ENSEMBLE
        [tkcurrent y_predictedkeep SHIFT] = fixTime(TESTINGCYCLE_DATA,n,pd,filtered,y(:,:,end),NUMBER_ENSEMBLE_MEMBERS);
        
    end
    %[tkcurrentkeep y_predictedkeep] = fixTime(TESTINGCYCLE_DATA,n,PREDICTION_INTERVAL,filtered,y,NUMBER_ENSEMBLE_MEMBERS)
    
    %tcurrent = t+15/60*forecastSteps;
    %tkcurrent(:,:,forecastSteps) = tcurrent; %+ 24 *(1+floor(forecastSteps/96));
    
    
    %thermalsunriseFunction(x(:,i,j-1),y(2,i),t(i),tHorizon(i))
    %tkcurrent = fixTime()
    %y_predicted(forecastSteps,:,:) = ensembleforecast(mean(xp,3),y_extended(1),tkcurrent(:,:,forecastSteps)); %CORRECT y_extended
    %ypredicted(forecastSteps,:,:) = mean(); %Ensemble
    
    ypredictedMean = mean(y_predictedkeep,3); %Ensemble
    
    for forecastSteps = 1:pd
        y_predicted(:,:,forecastSteps) = reshape(ypredictedMean(forecastSteps,:),size(y,1),size(y,2));
    end
    %y_predicted [pd x Pixels x Ensemble size]
    
    %end
    
    
    
    
    
    
    for le = 1:size(tkcurrent,3)
        tk(:,:,timestamp+le) = tkcurrent(:,:,le);
        ykeep(:,:,timestamp+le) = ynewkeep(:,:,le);
        %predictedObservation(:,:,timestamp+le)= y_predicted(:,:,le);
        y_predictedKeep(:,:,timestamp+le) = y_predicted;
    end
    
    %==============
    %==============
    %==============
    
    
    
    
    
    %[KeepFireDetect x_ prediction Keep_Q1 Keep_Inn Keep_S stResidual y cycle_out_scope cycle_reported estimation Keep_Q2] = stdsirDECONV4(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,prediction,Keep_Q1,Keep_Inn,Keep_S,KeepFireDetect,0,[],Keep_Q2);
    %[KeepFireDetect x_ Keep_Q1 Keep_Inn stResidual Keep_Q2] = stdsirDECONV5(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,Keep_Q1,Keep_Inn,KeepFireDetect,0,[],Keep_Q2);
    %
    
    
    %     [KeepFireDetect x_ Kn_ KeepAlpha stResidual y cycle_out_scope cycle_reported filtered Gf] = stdxtkalmanDECONV5(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,Kn_, KeepAlpha,KeepFireDetect,0,[]);
    %     [KeepFireDetect x_ Kn_ KeepAlpha stResidual y cycle_out_scope cycle_reported filtered]    = stdxtkalmanDECONV4(TESTINGCYCLE_DATA,NUMBER_STANDARD_DEVIATION,timestamp,n,x_,Kn_, KeepAlpha,KeepFireDetect);
    %     [fireTime cycle_reported standardResidual] = stdsirDECONV(TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude, longitude,column, row,NUMBER_STANDARD_DEVIATION,To_M, Ta_M, tm_M, ts_M, dT_M, Q, R1, bA, MNs, mp, cycle_out_scope);
    %     [fireTime cycle_reported standardResidual] = stdsir_ALTDECONV(TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude, longitude,column, row,NUMBER_STANDARD_DEVIATION,To_M, Ta_M, tm_M, ts_M, dT_M, Q, R1, bA, MNs, mp, cycle_out_scope);
    
    FLAG_BLURRED_PIXEL_NOTCONSIDERED(:,:,timestamp) = ones(size(KeepFireDetect,1),size(KeepFireDetect,2));
    
    %cycle_out_scope = 0:considered, 1:not considered In this code as
    %processed parallel, the cycle_out_scope is treated differently from
    %the code processing serial.
    %CONSIDER HOW TO PROCESS A PIXEL IF ITS NEIGHBOUR WAS NOT PREDICTED
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     prediction(:,:,timestamp) = x_(:,:,timestamp);
    %processPrediction = prediction(:,:,timestamp);
    %processPrediction(cycle_out_scope==1) = NaN; %I HAVE NOW FEW DATA, I WOULD LIKE TO ACCEPT THE USE OF TRAINING CYCLES LESS THAN 7, BUT TRY TO CONSIDER IF NO CYCLE IS PRESENT TO TRAIN AND IN THIS CASE ALL PARAMETERS ARE NaN. I AM STARTING FROM 31-07 TO TEST, AND I HAVE ONLY DATA FROM 23-07 TO TRAIN
    %ASSUMPTION THAT THERE WILL ALWAYS BE ENOUGH CYCLES TO TRAIN (7 CYCLES
    %ALWAYS). IN THIS CASE, WE USE THE FEW WE HAVE AT START UP (ON 31-07)
    %BUT AS WE GO ON, THE NUMBER OF TRAINING CYCLES INCREASE. PROBLEM IS
    %ONLY WHEN THERE IS EVEN NO CYCLE TO TRAIN.
    %prediction(:,:,timestamp) = processPrediction;
    
    KeepModelled(:,:,timestamp) = cycle_out_scope; %1=Non Modelled(lack of enough cycles (7) to train), 0=Modelled (7 cycles used to train to get the current DTC model)
    
    %         observation(:,:,timestamp) = y;
    
    KeepScaledResidual(:,:,timestamp) = stResidual;
    
    
    %n = n + 1;
    % fireTime CAN ALSO BE RETURNED FOR REPORT OR IT CAN BE FOUND FROM THIS FILE
    % REPORT POTENTIAL FIRE PIXELS
    % REPORT UNDECIDED DUE TO LACK OF CYCLES
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     n = n+1;
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % end
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % for timestamp = 1:NUMBER_TIMESTAMP
    %DECONVOLUTION STEP
    %START DECONVOLUTION
    
    
    
    
    %%%
    %%%
    
    
    
    
    %%%Case of noiseless dynamic model
    % % % % %         [x_ y t] = sequential_v(TESTINGCYCLE_DATA, timestamp,n,x_);
    % % % % %         KEEP_MOD_SEQ(:,:,timestamp) = x_(:,:,timestamp);
    % % % % %         n = n + 1;
    % % % % %         %plot(timestamp,x_(timestamp),'r-')
    
    if RESIDUAL == 0 %Prediction Residual
        [FIREFLAG PR] = predictionResidual(ynewkeep, y_predicted,THRESHOLD_POTENTIAL);
        KeepFireDetect(:,:,timestamp+le) = FIREFLAG;
        PRkeep(:,:,1+START_SIMULATION) = FEATURE1;
        PRkeep(:,:,timestamp+le) = PR;
        
    elseif RESIDUAL == 1  %OL
        %[FIREFLAG OL] = OLResidual(ynewkeep, Keep_Q2, y_predictedkeep, y_predicted, threshold); %signed OL
        [FIREFLAG OL] = ssrOLResidual(ynewkeep, Keep_Q2, y_predictedkeep, y_predicted, THRESHOLD_POTENTIAL); %The standard OL
        %squeeze(ynewkeep)
        %squeeze(Keep_Q2(:,:,timestamp))
        %mean(y_predictedkeep,3)
        %squeeze(y_predicted)
        KeepFireDetect(:,:,timestamp+le) = FIREFLAG;
        OLkeep(:,:,1+START_SIMULATION) = FEATURE1;
        OLkeep(:,:,timestamp+le) = OL;
        
    elseif RESIDUAL == 2  %gELL
        timestampp = timestamp
        %[FIREFLAG gELL] = ELLResidual(ynewkeep, Keep_Q2, y_predictedkeep, y_predicted, filtered, tkcurrent, threshold);
        [FIREFLAG gELL] = ssrELLResidual(ynewkeep, Keep_Q2, y_predictedkeep, y_predicted, filtered, tkcurrent, THRESHOLD_POTENTIAL); %The standard ELL
        KeepFireDetect(:,:,timestamp+le) = FIREFLAG;
        gELLkeep(:,:,1+START_SIMULATION) = FEATURE1;
        gELLkeep(:,:,timestamp+le) = gELL;
        
    else
        display('Specify residual (feature)')
    end
    
    n = n + 1;
    [EXPONENT timestamp]
end
%plot(squeeze(KEEP_MOD_SEQ(1,1,:)),'r')
%figure
%plot(TESTINGCYCLE_DATA(1,1).TEST_CYCLE_DATA,'b')
%hold off


% figure(20)
% plot(KEEP_MOD,'b')
% hold on
% plot(KEEP_MOD_SEQ,'r')
% title('sequential model')
% hold off


%ROW
%COLUMN
%TEST_CYCLE_TIME
%RESIDUAL
%FCTCOUNT_A
%ITERATIONS_A
%RMSE_SQ
%MAE_SQ
%BIAS_SQ

%     AA = sprintf('FIRERESULT%d',EXPONENT);
%     save(AA,'KeepResult')

%     AA = sprintf('FIRERESULTEnKF%d',EXPONENT*10);
%     save(AA,'KeepFireDetect')
%     AA = sprintf('FIRERESULTSIRda%d',EXPONENT*10);
%     save(AA,'KeepFireDetect')
% AA = sprintf('FIRERESULT4DVar%d',EXPONENT*10);
% save(AA,'KeepFireDetect')

%
%     BB = sprintf('FLAG_BLURRED_PIXEL_NOTCONSIDERED%d',EXPONENT);
%     save(BB,'FLAG_BLURRED_PIXEL_NOTCONSIDERED')

%     BB = sprintf('FLAG_BLURRED_PIXEL_NOTCONSIDEREDEnKF%d',EXPONENT*10);
%     save(BB,'FLAG_BLURRED_PIXEL_NOTCONSIDERED')
%     BB = sprintf('FLAG_BLURRED_PIXEL_NOTCONSIDEREDSIRda%d',EXPONENT*10);
%     save(BB,'FLAG_BLURRED_PIXEL_NOTCONSIDERED')
% BB = sprintf('FLAG_BLURRED_PIXEL_NOTCONSIDERED4DVar%d',EXPONENT*10);
% save(BB,'FLAG_BLURRED_PIXEL_NOTCONSIDERED')

% CC = sprintf('PRkeep4DVar%d',EXPONENT*10);
% save(CC,'PRkeep')


% DD = sprintf('observation4DVar%d',EXPONENT*10);
% save(DD,'ykeep')

% end
disp('finished prediction')
randn('state',sa);
rand('state',sb);

%save EnKFResultsAll
%save SIRdaResultsAll
% save w4DVarResultsAll


disp('finished')



%pause


%display('Pausing...')
%========
%========
%========

% % % % % % % % % % % % % % % % % % % % % % % % % % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % if FILTER==0 | FILTER==2
% % % % % % % % % % % % % % % % % % % % % % % % % %     predictedObservation(end)=[];
% % % % % % % % % % % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % % % % % % % % % % % % % % plot(squeeze(tk)+ [(8-8)*24*ones(1,95) (9-8)*24*ones(1,96) (10-8)*24*ones(1,96) (11-8)*24*ones(1,96+END_SIMULATION)].',squeeze(ykeep),'b*')
% % % % % % % % % % % % % % % % % % % % % % % % % % hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % %stairs(squeeze(tk(:,:,1:timestamp)),squeeze(predictedObservation),'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % %stairs(squeeze(tkcurrent),squeeze(mean(y_predicted,3)),'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % stairs( [squeeze(tk(:,:,1:timestamp));squeeze(tkcurrent) + [(8-8)*24*ones(1,-END_SIMULATION+SHIFT(:,:,1)) (9-8)*24*ones(1,96-SHIFT(:,:,1)+SHIFT(:,:,2)) (10-8)*24*ones(1,96-SHIFT(:,:,2)+SHIFT(:,:,3)) (11-8)*24*ones(1,96+END_SIMULATION-SHIFT(:,:,3))].'],[squeeze(predictedObservation); squeeze(y_predicted)],'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % figure

%tk
%tkcurrent

%ykeep
%ynewkeep

%predictedObservation(end)=[];
%y_predicted

%========
%========
%========




% pause

%KEEP_MOD_SEQ = prediction;
KEEP_MOD_SEQ = y_predictedKeep; %predictedObservation;

for i = INDEX_ROW %1:size(TESTINGCYCLE_DATA,1)
    for j = INDEX_COLUMN %1:size(TESTINGCYCLE_DATA,2)
        
        NUMBER_TIMESTAMP = startIndex(i,j)+(SQ_LENGTH-1)-1; %sum(TESTINGCYCLE_DATA(i,j).TEST_CYCLE_LENGTH(1:5)); %300; %Only 5 cycles will be taken into account.
        ROW = TESTINGCYCLE_DATA(i,j).ROW(1);
        COLUMN = TESTINGCYCLE_DATA(i,j).COLUMN(1);
        
        TEST_CYCLE_TIME = TESTINGCYCLE_DATA(i,j).TEST_CYCLE_TIME(startIndex(i,j):NUMBER_TIMESTAMP);
        %RESIDUAL = squeeze(KEEP_MOD_SEQ(i,j,1:NUMBER_TIMESTAMP-startIndex(i,j)+1)).' - TESTINGCYCLE_DATA(i,j).TEST_CYCLE_DATA(startIndex(i,j):NUMBER_TIMESTAMP);
        
        
        if FILTER==2
            %RESIDUAL = squeeze(KEEP_MOD_SEQ(i,j,1:NUMBER_TIMESTAMP-ASSIMILATION_WINDOW_LENGTH-startIndex(i,j)+1+pd)).' - TESTINGCYCLE_DATA(i,j).TEST_CYCLE_DATA(startIndex(i,j):NUMBER_TIMESTAMP-ASSIMILATION_WINDOW_LENGTH+pd);
            RESIDUAL = squeeze(KEEP_MOD_SEQ(i,j,1:NUMBER_TIMESTAMP-startIndex(i,j)+1+pd)).' - TESTINGCYCLE_DATA(i,j).TEST_CYCLE_DATA(startIndex(i,j):NUMBER_TIMESTAMP+pd);
        else
            RESIDUAL = squeeze(KEEP_MOD_SEQ(i,j,1:NUMBER_TIMESTAMP-startIndex(i,j)+1+pd)).' - TESTINGCYCLE_DATA(i,j).TEST_CYCLE_DATA(startIndex(i,j):NUMBER_TIMESTAMP+pd);
        end
        %x(n) = A x(n-1) + v(n)
        %RESIDUAL = squeeze(KEEP_MOD_SEQ(i,j,1:NUMBER_TIMESTAMP-startIndex(i,j))).' - TESTINGCYCLE_DATA(i,j).TEST_CYCLE_DATA(startIndex(i,j):NUMBER_TIMESTAMP-1);
        
        
        FCTCOUNT_A = 1;
        ITERATIONS_A = 1;
        
        
        RMSE_SQ = sqrt(mean(RESIDUAL.^2));
        MAE_SQ = mean(abs(RESIDUAL));
        BIAS_SQ = sum(RESIDUAL);
        
        %Give none due to the fact that I started from cycle 9, then cycle
        %8, give 0
        %RESIDUAL_1ST = squeeze(KEEP_MOD_SEQ(i,j,startIndex(i,j):TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_LENGTH(1))).' - TESTINGCYCLE_DATA(i,j).TEST_CYCLE_DATA(startIndex(i,j):TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_LENGTH(1));
        %CHANGE THESE
        %RESIDUAL_1ST = squeeze(KEEP_MOD_SEQ(i,j,startIndex(i,j):TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_LENGTH(1))).' - TESTINGCYCLE_DATA(i,j).TEST_CYCLE_DATA(startIndex(i,j):TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_LENGTH(1));
        
        if FILTER==2
            %RESIDUAL_1ST = squeeze(KEEP_MOD_SEQ(i,j,startIndex(i,j)-1:TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_LENGTH(1)-1-ASSIMILATION_WINDOW_LENGTH)).' - TESTINGCYCLE_DATA(i,j).TEST_CYCLE_DATA(startIndex(i,j):TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_LENGTH(1)-ASSIMILATION_WINDOW_LENGTH);
            RESIDUAL_1ST = squeeze(KEEP_MOD_SEQ(i,j,startIndex(i,j):TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_LENGTH(1))).' - TESTINGCYCLE_DATA(i,j).TEST_CYCLE_DATA(startIndex(i,j):TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_LENGTH(1));
        else
            %RESIDUAL_1ST = squeeze(KEEP_MOD_SEQ(i,j,startIndex(i,j)-1:TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_LENGTH(1)-1)).' - TESTINGCYCLE_DATA(i,j).TEST_CYCLE_DATA(startIndex(i,j):TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_LENGTH(1));
            RESIDUAL_1ST = squeeze(KEEP_MOD_SEQ(i,j,startIndex(i,j):TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_LENGTH(1))).' - TESTINGCYCLE_DATA(i,j).TEST_CYCLE_DATA(startIndex(i,j):TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_LENGTH(1));
        end
        
        RMSE_SQ_1ST = sqrt(mean(RESIDUAL_1ST.^2));
        MAE_SQ_1ST = mean(abs(RESIDUAL_1ST));
        BIAS_SQ_1ST = sum(RESIDUAL_1ST);
        
        
        %SSE_SQ(i,j) = sum((squeeze(KEEP_MOD_SEQ(i,j,:))' - TESTINGCYCLE_DATA(i,j).TEST_CYCLE_DATA(1:576)).^2);
        
        
        %Report
        %             for B = 1:3
        %
        %                 if B*200> NUMBER_TIMESTAMP - startIndex(i,j) + 1
        %
        %                     JTIME = TEST_CYCLE_TIME((B-1)*200+1:end);
        %                     JRESIDUAL = RESIDUAL((B-1)*200+1:end);
        %
        %                 else
        %                     JTIME = TEST_CYCLE_TIME((B-1)*200+1:B*200);
        %                     JRESIDUAL = RESIDUAL((B-1)*200+1:B*200);
        %
        %                 end
        %
        %
        %
        %
        %                 %[XLS_ROW_NUMBER_POTENTIAL FILE_NUMBER_POTENTIAL] = resultReport2(FILENAMEPART,KeepFireDetect(:,:,timestamp),timestamp,n,y,stResidual,THRESHOLD_POTENTIAL,EXPONENT,1,XLS_ROW_NUMBER_POTENTIAL,FILE_NUMBER_POTENTIAL,EXCEL,Excel_Potential);
        %                 [XLS_ROW_NUMBER_POTENTIAL FILE_NUMBER_POTENTIAL] = resultReport5(FILENAMEPART, ROW, COLUMN, JTIME, JRESIDUAL, FCTCOUNT_A, ITERATIONS_A, RMSE_SQ, MAE_SQ, BIAS_SQ,XLS_ROW_NUMBER_POTENTIAL,FILE_NUMBER_POTENTIAL,EXCEL,Excel_Potential);
        %                 %
        %
        %                 XLS_ROW_NUMBER_POTENTIAL = XLS_ROW_NUMBER_POTENTIAL +1;
        %
        %             end
        
        %             XLS_ROW_NUMBER_POTENTIAL = XLS_ROW_NUMBER_POTENTIAL +2;
        
        
        [ROW COLUMN]
        
    end
end


TIME = TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_TIME;
OBSERVATION = TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA(startIndex(INDEX_ROW,INDEX_COLUMN):NUMBER_TIMESTAMP+pd); %ykeep
if FILTER == 2
    %PREDICTION = squeeze(KEEP_MOD_SEQ(INDEX_ROW,INDEX_COLUMN,1:NUMBER_TIMESTAMP-startIndex(INDEX_ROW,INDEX_COLUMN)+1-ASSIMILATION_WINDOW_LENGTH+pd));
    PREDICTION = squeeze(KEEP_MOD_SEQ(INDEX_ROW,INDEX_COLUMN,1:NUMBER_TIMESTAMP-startIndex(INDEX_ROW,INDEX_COLUMN)+1+pd));
else
    PREDICTION = squeeze(KEEP_MOD_SEQ(INDEX_ROW,INDEX_COLUMN,1:NUMBER_TIMESTAMP-startIndex(INDEX_ROW,INDEX_COLUMN)+1+pd));
end
%x(n) = A x(n-1) + v(n)
%PREDICTION = squeeze(KEEP_MOD_SEQ(INDEX_ROW,INDEX_COLUMN,1:NUMBER_TIMESTAMP-startIndex(INDEX_ROW,INDEX_COLUMN)));

startIndex(INDEX_ROW,INDEX_COLUMN)



addpath('C:\Users\JOSINE\WORK1\work\FireDetection2\COMPARE_FIR_MODIS')
load MODISfile
load FIRfileCorrected

MM = MODIS(CENTRE_ROW-778+1,CENTRE_COLUMN-822+1)
FF = FIR(CENTRE_ROW-778+1,CENTRE_COLUMN-822+1)


%save SQ_BER_6CYCLES KEEP_TIME TIME OBSERVATION PREDICTION RESIDUAL RMSE_SQ MAE_SQ BIAS_SQ RMSE_SQ_1ST MAE_SQ_1ST BIAS_SQ_1ST
%save SQ_BER_3CYCLES_DYNAMIC_OWN_SIR KEEP_TIME TIME OBSERVATION PREDICTION RESIDUAL RMSE_SQ MAE_SQ BIAS_SQ %RMSE_SQ_1ST MAE_SQ_1ST BIAS_SQ_1ST
%save SQ_BER_FIRESIMULATIONDAY_SIR KEEP_TIME TIME OBSERVATION PREDICTION RESIDUAL RMSE_SQ MAE_SQ BIAS_SQ MM FF KeepFireDetect %RMSE_SQ_1ST MAE_SQ_1ST BIAS_SQ_1ST
%save SQ_BER_FIRESIMULATIONMORNING_XTKALMAN KEEP_TIME TIME OBSERVATION PREDICTION RESIDUAL RMSE_SQ MAE_SQ BIAS_SQ %RMSE_SQ_1ST MAE_SQ_1ST BIAS_SQ_1ST


[RMSE_SQ MAE_SQ BIAS_SQ RMSE_SQ_1ST MAE_SQ_1ST BIAS_SQ_1ST]

if length(TESTINGCYCLE_DATA.TEST_CYCLE_LENGTH)>2
    RANGE_DISPLAY = sum(TESTINGCYCLE_DATA.TEST_CYCLE_LENGTH(1:end-2))+1:sum(TESTINGCYCLE_DATA.TEST_CYCLE_LENGTH(1:end));
else
    RANGE_DISPLAY = startIndex:sum(TESTINGCYCLE_DATA.TEST_CYCLE_LENGTH(1:end));
end

figure
%plot(KEEP_TIME(startIndex(INDEX_ROW,INDEX_COLUMN):end),TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA(startIndex(INDEX_ROW,INDEX_COLUMN):NUMBER_TIMESTAMP),'kx')
plot(KEEP_TIME(RANGE_DISPLAY),TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA(RANGE_DISPLAY),'kx')

hold on
%stairs(KEEP_TIME(startIndex(INDEX_ROW,INDEX_COLUMN):end),squeeze(KEEP_MOD_SEQ(INDEX_ROW,INDEX_COLUMN,1:NUMBER_TIMESTAMP-startIndex(INDEX_ROW,INDEX_COLUMN)+1)),'r')
if FILTER==2
    stairs(KEEP_TIME(RANGE_DISPLAY(1:end-ASSIMILATION_WINDOW_LENGTH)),squeeze(KEEP_MOD_SEQ(INDEX_ROW,INDEX_COLUMN,RANGE_DISPLAY(1:end-ASSIMILATION_WINDOW_LENGTH)-startIndex+1)),'r')
else
    stairs(KEEP_TIME(RANGE_DISPLAY),squeeze(KEEP_MOD_SEQ(INDEX_ROW,INDEX_COLUMN,RANGE_DISPLAY-startIndex+1)),'r')
end
%x(n) = A x(n-1) + v(n)
%plot(KEEP_TIME(1:575),squeeze(KEEP_MOD_SEQ(INDEX_ROW,INDEX_COLUMN,1:NUMBER_TIMESTAMP-startIndex(INDEX_ROW,INDEX_COLUMN))),'r')

%plot(KEEP_TIME(startIndex(INDEX_ROW,INDEX_COLUMN):end),280 + 10*KeepFireDetect(:),'m')

if FILTER==2
    %plot(KEEP_TIME(RANGE_DISPLAY(1:end-ASSIMILATION_WINDOW_LENGTH)),280 + 10*squeeze(KeepFireDetect(INDEX_ROW,INDEX_COLUMN,RANGE_DISPLAY(1:end-ASSIMILATION_WINDOW_LENGTH)-startIndex+1)),'m')
    plot(KEEP_TIME(RANGE_DISPLAY(1:end)),280 + 10*squeeze(KeepFireDetect(INDEX_ROW,INDEX_COLUMN,RANGE_DISPLAY(1:end)-startIndex+1)),'m')
else
    plot(KEEP_TIME(RANGE_DISPLAY),280 + 10*squeeze(KeepFireDetect(INDEX_ROW,INDEX_COLUMN,RANGE_DISPLAY-startIndex+1)),'m')
end
text(KEEP_TIME(end) + 30/60,290,'fire')
text(KEEP_TIME(end) + 30/60,280,'no fire')

legend('Observed temperature','Predicted temperature','condition = (fire, no fire)')

hold off

%figure

LAUNCH_TIME = TESTINGCYCLE_DATA.TEST_CYCLE_TIME(startIndex);

%save CFAR_Example4CFAR KEEP_TIME TIME OBSERVATION PREDICTION RESIDUAL RMSE_SQ MAE_SQ BIAS_SQ MM FF KeepFireDetect %RMSE_SQ_1ST MAE_SQ_1ST BIAS_SQ_1ST
%save CFAR_Example25CFAR KEEP_TIME TIME OBSERVATION PREDICTION RESIDUAL RMSE_SQ MAE_SQ BIAS_SQ MM FF KeepFireDetect %RMSE_SQ_1ST MAE_SQ_1ST BIAS_SQ_1ST

%save ExampleEnKF29 KEEP_TIME TIME OBSERVATION PREDICTION RESIDUAL RMSE_SQ MAE_SQ BIAS_SQ MM FF KeepFireDetect TESTINGCYCLE_DATA KEEP_MOD_SEQ RANGE_DISPLAY startIndex LAUNCH_TIME %RMSE_SQ_1ST MAE_SQ_1ST BIAS_SQ_1ST
%save ExampleSIRda29 KEEP_TIME TIME OBSERVATION PREDICTION RESIDUAL RMSE_SQ MAE_SQ BIAS_SQ MM FF KeepFireDetect TESTINGCYCLE_DATA KEEP_MOD_SEQ RANGE_DISPLAY startIndex LAUNCH_TIME %RMSE_SQ_1ST MAE_SQ_1ST BIAS_SQ_1ST
%save Example4DVar3 KEEP_TIME TIME OBSERVATION PREDICTION RESIDUAL RMSE_SQ MAE_SQ BIAS_SQ MM FF KeepFireDetect TESTINGCYCLE_DATA KEEP_MOD_SEQ RANGE_DISPLAY startIndex LAUNCH_TIME %RMSE_SQ_1ST MAE_SQ_1ST BIAS_SQ_1ST


%[cycle ROW COLUMN]

%figure
%plot(RESIDUAL)




%end

%hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % clear FNDATAREGION
%save TESTING
%close all

% figure
%
% plot(KEEP_MOD,'r')
% hold on
% plot(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA,'b')
% hold off
% SSE_NSQ = sum((KEEP_MOD - TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA).^2)
% RMSE_NSQ = sqrt(mean((KEEP_MOD - TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA).^2))
% MAD_NSQ = mean(abs(KEEP_MOD - TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA))



%%%%%%%%%%%%%%%%%%%%%


%IMPLEMENT CFAR DETECTION
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % NUMBER_STANDARD_DEVIATION = 1; %THRESHOLD CONSIDERING CFAR
%DETECTION_METHOD = 1; %1 Using residual test statistic, 2 Using an alternative method
%stdkalman(MOD,TEST_FIRE_CYCLE,AVERAGE*(1-round(E(TEST_FIRE_CYCLE_NUMBER))), TEST_FIRE_CYCLE_NUMBER);
%stdsir(TEST_FIRE_CYCLE,To,Ta,w,tm,ts,k,dT,AVERAGE*(1-round(E(TEST_FIRE_CYCLE_NUMBER))),TEST_FIRE_CYCLE_NUMBER);


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             LENGTH_CURRENT_CYCLE = length(TEST_FIRE_CYCLE);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             %CURRENT_CYLE = TEST_FIRE_CYCLE_NUMBER;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             [To_M Ta_M tm_M ts_M dT_M Q R1 bA MNs mp cycle_out_scope] = noisemodel(current_txt_file,TEST_FIRE_CYCLE_NUMBER,latitude, longitude,LENGTH_CURRENT_CYCLE);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             %             %REPORT UNDECIDED



% sa = randn('state');
% randn('state',0)



%SET STARTING TIME OF DETECTION AND FIND THE INDEX TO START FOR EACH
%PIXEL--DETECTION HAS TO START AT SAME TIME, NOT AT SAME SAMPLE FROM THE
%HEAT RISE.

%Check the length of time is equal for all pixels
% CHECK_LENGTH = [];
% [NUMBER_ROW NUMBER_COLUMN] = size(TESTINGCYCLE_DATA);
% for iN= 1:NUMBER_COLUMN
%     for jN = 1:NUMBER_ROW
%         CHECK_LENGTH = [CHECK_LENGTH length(TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME)]; %CAN BE DISPLAYED ON COMMAND WINDOW TO CHECK NUMBER OF SAMPLES FOR EACH PIXEL
%     end
% end
%
% if(sum(CHECK_LENGTH == CHECK_LENGTH(1))==NUMBER_COLUMN*NUMBER_ROW)
%     display('time length is the same for all pixels');
% end
%
% %Check the first time stamp
% CHECK_TIME = [];
% for iN= 1:NUMBER_COLUMN
%     for jN = 1:NUMBER_ROW
%         CHECK_TIME(jN,iN) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME(1);
%     end
% end
%
% [iCT jCT] = max(CHECK_TIME);
%
% for iN= 1:NUMBER_COLUMN
%     for jN = 1:NUMBER_ROW
%         TESTINGCYCLE_DATA(jN,iN).START_INDEX = find(TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME(1:48)==iCT(1));
%     end
% end
%
%
% %The starting indices for each pixels
% n = zeros(NUMBER_ROW,NUMBER_COLUMN);
% for jN = 1:NUMBER_ROW
%     for iN = 1:NUMBER_COLUMN
%         n(jN,iN) = TESTINGCYCLE_DATA(jN,iN).START_INDEX;
%     end
% end
%
% % if cycle_out_scope==0
%
% x_ = [];
%
% NUMBER_TIMESTAMP = 300;
%
%
%
% %%%%%%%%%%%%%%%%%%%%%
%
%
% n = 1;
% KEEP_MOD_SEQ = [];
% figure(10)
% hold on
% for timestamp = 1:length(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA)
%     [x_ y t] = sequential_v(TESTINGCYCLE_DATA, timestamp,n,x_);
%     KEEP_MOD_SEQ = [KEEP_MOD_SEQ x_(timestamp)];
%     n = n + 1;
%     %plot(timestamp,x_(timestamp),'r-')
% end
% plot(KEEP_MOD_SEQ,'r')
% plot(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA,'b')
% hold off
%
%
% figure(20)
% plot(KEEP_MOD,'b')
% hold on
% plot(KEEP_MOD_SEQ,'r')
% title('sequential model')
% hold off
%
%
% SSE_SQ = sum((KEEP_MOD_SEQ - TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA).^2)
% RMSE_SQ = sqrt(mean((KEEP_MOD_SEQ - TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA).^2))
% MAD_SQ = mean(abs(KEEP_MOD_SEQ - TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA))
%
%
%
% %close all
% for k = 2:6
%     figure
%
%     startSample = KEEP_START(k);
%     if k<6
%         endSample = KEEP_START(k+1)-1;
%     else
%         endSample = length(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA);
%     end
%     sunriseSample = KEEP_SUNRISE(k);
%     sunsetSample = KEEP_SUNSET(k);
%
%     subplot(2,1,1)
%     plot(KEEP_tSTART(k):15/60:KEEP_tSTART(k)+(length(KEEP_MOD(startSample:endSample))-1)*15/60,KEEP_MOD(startSample:endSample),'r')
%     hold on
%     plot(KEEP_tSTART(k):15/60:KEEP_tSTART(k)+(length(KEEP_MOD(startSample:endSample))-1)*15/60,TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA(startSample:endSample),'b')
%     %axis([KEEP_tSTART(k) KEEP_tSTART(k)+(length(KEEP_MOD(startSample:endSample))-1)*15/60 260 310])
%     axis([0 24-15/60 260 310])
%     %%%%%%%%%%%%%%%%%%%
%     hold off
%
%     %figure
%     subplot(2,1,2)
%     theta_z = -90:90;
%     tau = 0.03; %Clear sky
%
%     %m = 1./cosd(theta_z); %Simple
%     RE = 6371000;
%     H = 8430;
%     m = -RE/H*cosd(theta_z) + sqrt((RE/H*cosd(theta_z)).^2 + 2*RE/H + 1);
%
%     IL = 3; %This value only for scaling purposes
%     I0 = 1367*IL;%
%     eN = 0.016704;
%     Iext = I0*eN*cosd(theta_z).*exp(-tau*m);
%
%
%     plot(theta_z*KEEP_tDAYLENGTH(k)/180 + KEEP_tSOLARNOON(k),Iext+200)
%     hold on
%
%     plot(0:15/60/100:theta_z(1)*KEEP_tDAYLENGTH(k)/180 + KEEP_tSOLARNOON(k), zeros(1,length(0:15/60/100:theta_z*KEEP_tDAYLENGTH(k)/180 + KEEP_tSOLARNOON(k)))+min(Iext+200))
%     plot(theta_z(end)*KEEP_tDAYLENGTH(k)/180 + KEEP_tSOLARNOON(k):15/60/100:24-15/60, zeros(1,length(theta_z(end)*KEEP_tDAYLENGTH(k)/180 + KEEP_tSOLARNOON(k):15/60/100:24-15/60))+min(Iext+200))
%     plot(KEEP_tSTART(k):15/60:KEEP_tSTART(k)+(length(KEEP_MOD(startSample:endSample))-1)*15/60,KEEP_MOD(startSample:endSample) - min(KEEP_MOD(startSample:endSample)) + 200,'r')
%
%     %%%%%%%%%%%%%%%%%%%%
%
%     %axis([KEEP_tSTART(k) KEEP_tSTART(k)+(length(KEEP_MOD(startSample:endSample))-1)*15/60 190 250])
%     axis([0 24-15/60 190 270])
%
%     hold off
% end
%
%
%
% %%%%%%%%%%%%%%
%
% %close all
% figure
%
%
%
% startSample = KEEP_START(6);
% endSample = length(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA);
%
% KEEP_MOD(startSample+20) = KEEP_MOD(startSample+19);
% KEEP_MOD(startSample+21) = KEEP_MOD(startSample+19);
%
% sunriseSample = KEEP_SUNRISE(6);
% sunsetSample = KEEP_SUNSET(6);
%
% subplot(2,1,1)
% plot(KEEP_tSTART(6):15/60:KEEP_tSTART(6)+(length(KEEP_MOD(startSample:endSample))-1)*15/60,KEEP_MOD(startSample:endSample),'r')
% %hold on
% %plot(KEEP_tSTART(6):15/60:KEEP_tSTART(6)+(length(KEEP_MOD(startSample:endSample))-1)*15/60,TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA(startSample:endSample),'b')
% %axis([KEEP_tSTART(k) KEEP_tSTART(k)+(length(KEEP_MOD(startSample:endSample))-1)*15/60 260 310])
% axis([0 24-15/60 260 310])
% %%%%%%%%%%%%%%%%%%%
% %hold off
%
% %figure
% subplot(2,1,2)
% theta_z = -90:90;
% tau = 0.03;%0.03; %Clear sky
%
% %m = 1./cosd(theta_z); %Simple
% RE = 6371000;
% H = 8430;
% m = -RE/H*cosd(theta_z) + sqrt((RE/H*cosd(theta_z)).^2 + 2*RE/H + 1);
%
% IL = 3; %This value only for scaling purposes
% I0 = 1367*IL;%
% eN = 0.016704;
% Iext = I0*eN*cosd(theta_z).*exp(-tau*m);
% %for theta_z = 0:90;
% tau = 0.9;
% Iext(theta_z>0) = I0*2.37*eN*cosd(theta_z(theta_z>0)).*exp(-tau*m(theta_z>0));
%
%
% plot(theta_z*KEEP_tDAYLENGTH(6)/180 + KEEP_tSOLARNOON(6),Iext+200)
% hold on
%
%
% plot(0:15/60/100:theta_z(1)*KEEP_tDAYLENGTH(6)/180 + KEEP_tSOLARNOON(6), zeros(1,length(0:15/60/100:theta_z*KEEP_tDAYLENGTH(6)/180 + KEEP_tSOLARNOON(6)))+min(Iext+200))
% plot(theta_z(end)*KEEP_tDAYLENGTH(6)/180 + KEEP_tSOLARNOON(6):15/60/100:24-15/60, zeros(1,length(theta_z(end)*KEEP_tDAYLENGTH(6)/180 + KEEP_tSOLARNOON(6):15/60/100:24-15/60))+min(Iext+200))
% plot(KEEP_tSTART(6):15/60:KEEP_tSTART(6)+(length(KEEP_MOD(startSample:endSample))-1)*15/60,KEEP_MOD(startSample:endSample) - min(KEEP_MOD(startSample:endSample)) + 210,'r')
%
% %%%%%%%%%%%%%%%%%%%%
%
% %axis([KEEP_tSTART(k) KEEP_tSTART(k)+(length(KEEP_MOD(startSample:endSample))-1)*15/60 190 250])
% axis([0 24-15/60 190 270])
%
% hold off

% Excel_Potential.Quit %Quit the application
% Excel_Potential.delete %Delete the object
% clear Excel_Potential

clear FNDATAREGION
%clear TESTINGCYCLE_DATA
randn('state',sa);
rand('state',sb);
disp('finished')