function [MOD,To_M Ta_M tm_M ts_M dT_M w_M k_M Qc Rc SAVE_TRAINING_CYCLE SAVE_NOISE_MEAS mp cycle_out_scope w2_M tau_M m_z_M P_M v_z_M v_m_M theta_z_M theta_zm_M theta_zs_M m_zs_M tsr_P_M a_P_M Y_P_M, Z_P_M a_RKHS_M K_RKHS_M deltaK_RKHS_M a_SVD_M U_SVD_M deltaU_SVD_M ITERATIONS_A FCTCOUNT_A  RMSE_NSQ_OVER_MISSING_VALUE_ONLY MAE_NSQ_OVER_MISSING_VALUE_ONLY BIAS_NSQ_OVER_MISSING_VALUE_ONLY tmin_M horizon missingdataFlag fParameter]= noisemodel(current_txt_file,current_cycle,latitude, longitude,length_current_cycle, MODEL,ErrFct,ACTIVATE_MISSINGSAMPLE,GAP,INDEX_ROW,INDEX_COLUMN,DDTP_Nbre_Cycles)
%function [MOD,To_M Ta_M tm_M ts_M dT_M w_M k_M Qc Rc SAVE_TRAINING_CYCLE SAVE_NOISE_MEAS mp cycle_out_scope w2_M tau_M m_z_M P_M v_z_M v_m_M theta_z_M theta_zm_M theta_zs_M m_zs_M tsr_P_M a_P_M Y_P_M, Z_P_M a_RKHS_M K_RKHS_M deltaK_RKHS_M a_SVD_M U_SVD_M deltaU_SVD_M ITERATIONS_A FCTCOUNT_A  RMSE_NSQ_OVER_MISSING_VALUE_ONLY MAE_NSQ_OVER_MISSING_VALUE_ONLY BIAS_NSQ_OVER_MISSING_VALUE_ONLY tmin_M horizon missingdataFlag]= noisemodel(current_txt_file,current_cycle,latitude, longitude,length_current_cycle, MODEL,ErrFct,ACTIVATE_MISSINGSAMPLE,GAP,INDEX_ROW,INDEX_COLUMN,DDTP_Nbre_Cycles)
%function [To_M Ta_M tm_M ts_M dT_M Qc Rc SAVE_TRAINING_CYCLE SAVE_NOISE_MEAS mp cycle_out_scope w_M]= noisemodel(current_txt_file,current_cycle,latitude, longitude,length_current_cycle)

missingdataFlag  = [];
% close all
% clear all
% clc
% sa = randn('state');
% randn('state',0);
%d = dir('msgcsv/pix_88806.coord');
%disp('latitude longitude')
%st = zeros(length(d),2);
%for i = 1:length(d)
%NUMBER_OF_CYCLE = 14;%===========================================================================================================
% NUMBER_OF_CYCLE = 22;
% %Number of cycles for variance training
% TRAIN_CYCLES = current_cycle-7:current_cycle-1;
%
%
%
%
%
% %TRAIN_CYCLES = [6 7 11 12 13 14];
% %TRAIN_CYCLES = 1:NUMBER_OF_CYCLE;
%
% %NUMBER_OF_CYCLE_USED_TO_TRAIN = length(TRAIN_CYCLES);
% NUMBER_OF_CYCLE_USED_TO_TRAIN = length(TRAIN_CYCLES);



% a = VARIANCE_TRAINING_CYCLES;

DATA = load(current_txt_file);
% % % % % %a = load('msgcsv/pix_88806.txt');
% % % % % %a = load('msgcsv/pix_88305.txt');
% % % % % %OVERLAP = 1;
% % % % % %OVERLAP = 1;
% % % % %
% % % % %
% % % % % file = current_txtfile; %'F:\MSG_DATA_2007\D_34_MSG\34004390.txt';
% % % % % file(length(file)-2:length(file)) = 'coo';
% % % % % filenew = strcat(file,'rd');
% % % % % fid = fopen(filenew,'r');
% % % % %
% % % % % % fid = fopen('msgcsv/pix_88305.coord','r');
% % % % %
% % % % %
% % % % %
% % % % %
% % % % % % a = load('msgcsv/pix_88806.txt');
% % % % % % fid = fopen('msgcsv/pix_88806.coord','r');
% % % % % % OVERLAP = 5;
% % % % %
% % % % %
% % % % % % s = fscanf(fid,'%f');
% % % % % % dcoord = [s(2)  s(3)];
% % % % % % LAT = abs(dcoord(1));
% % % % %
% % % % %
% % % % % for il = 1:4
% % % % %     s = fscanf(fid,'%s',1);
% % % % %     if il==2 | il==4
% % % % %         Al(1,il/2) = sscanf(s,'%f');
% % % % %     end
% % % % % end
% % % % %
% % % % % dcoord = Al;
% % % % %
% % % % %
% % % % % LAT_T = dcoord(1);
% % % % % LAT = abs(dcoord(1));
% % % % %
% % % % % fclose(fid);
% % % % % %end

% addpath('C:\Users\JOSINE\WORK1\work\FireDetection2\solarTimeLocation')
DATE = [2007 7 23;
    2007 7 24;
    2007 7 25;
    2007 7 26;
    2007 7 27;
    2007 7 28;
    2007 7 29;
    2007 7 30;
    2007 7 31;
    2007 8 1;
    2007 8 2;
    2007 8 3;
    2007 8 4;
    2007 8 5;
    2007 8 6;
    2007 8 7;
    2007 8 8;
    2007 8 9;
    2007 8 10;
    2007 8 11;
    2007 8 12;
    2007 8 13;
    2007 8 14];


daySunriseL = zeros(1,size(DATE,1));
daySunriseR = zeros(1,size(DATE,1));
daySolarnoon = zeros(1,size(DATE,1));
daySunset = zeros(1,size(DATE,1));
dayDaylength = zeros(1,size(DATE,1));
daySunrise90 = zeros(1,size(DATE,1)); %Return 0 for 90 sunrise, the case of other model except Inamdar_Duan_2013 and DDTP.


%Thermal sunrise from sun_rise_set(1) to sun_rise_set(3)
%ts initialization as sun_rise_set(2)
switch MODEL
    case {'z', 'j'} %Gottsche2009 and Jiang2006
        sun_rise_set = [96 96 84];
    case {'p', 's', 'g', 'v', 'i', 'd', 'r', 't', 'u'}
        %elseif (MODEL == 'p') || (MODEL == 's') || (MODEL == 'g') || (MODEL == 'j') || (MODEL == 'v') || (MODEL == 'i') || (MODEL == 'd')
        sun_rise_set = [84 96 84];

    otherwise
        disp('Thermal sun: Specify the right model');
        error('Specify the right model');
end



for iday = 1:size(DATE,1)
    year = DATE(iday,1);
    month = DATE(iday,2);
    day = DATE(iday,3);
    [sunriseL sunriseR solarnoon sunset daylength] = solartime2(sun_rise_set,year,month,day,longitude,latitude);
    daySunriseL(iday) = sunriseL;
    daySunriseR(iday) = sunriseR;
    daySolarnoon(iday) = solarnoon;
    daySunset(iday) = sunset;
    dayDaylength(iday) = daylength;
    %daySunrise90(iday) = 0; %Return 0 for 90 sunrise, the case of other model except Inamdar_Duan_2013 and DDTP.
    if MODEL == 'd' | MODEL == 'u'
        [sunrise90 sunrise90 solarnoon90 sunset90 daylength90] = solartime2([90 90 90],year,month,day,longitude,latitude);
        daySunrise90(iday) = sunrise90;
    end
end

%Change this time so that 23-07-2007 00:00 be the 0 hour. The time
%is expressed in hours.
clear sunriseL sunriseR solarnoon sunset
SUNRISEL = zeros(1,size(DATE,1));
SUNRISER = zeros(1,size(DATE,1));
solarnoon = zeros(1,size(DATE,1));
sunset = zeros(1,size(DATE,1));
morningHorizonTime = zeros(1,size(DATE,1));

for iday = 1:size(DATE,1)
    SUNRISEL(iday) = (iday-1)*24 + daySunriseL(iday);
    SUNRISER(iday) = (iday-1)*24 + daySunriseR(iday);
    solarnoon(iday) = (iday-1)*24 + daySolarnoon(iday);
    sunset(iday) = (iday-1)*24 + daySunset(iday);
    if MODEL == 'u'
        morningHorizonTime(iday) = (iday-1)*24 + daySunrise90(iday);
    end
end


%SAMPLE = DATA(:,1); %In a file it starts from 0 to 2207 %3534212172:15:3534233757; %Time starts from 00:12 for the first day (23-07-2007) and end at 23:57 on the last day (14-08-2007)() %a(:,1)
TIME = (0+12:15:2207*15+12)/60; %Time taken from 23-07-2007 00:00. This time is 0. The time is expressed in hours.
BRIGHTNESS = DATA(:,2).';


iSunrise = zeros(1,size(DATE,1));
iSunriseL = zeros(1,size(DATE,1));
iSunriseR = zeros(1,size(DATE,1));

%Find the sunrise point samples
for iday = 1:size(DATE,1)
    iBeforeSunriseL = find(TIME<=SUNRISEL(iday));
    sBeforeSunriseL = [iBeforeSunriseL.' TIME(iBeforeSunriseL).'];
    tBeforeSunriseL  = sBeforeSunriseL(:,2);
    [dfBeforeSunriseL ifBeforeSunriseL] = max(tBeforeSunriseL);
    iSunriseL(iday) = sBeforeSunriseL(ifBeforeSunriseL,1);

    iBeforeSunriseR = find(TIME<SUNRISER(iday));
    sBeforeSunriseR = [iBeforeSunriseR.' TIME(iBeforeSunriseR).'];
    tBeforeSunriseR  = sBeforeSunriseR(:,2);
    [dfBeforeSunriseR ifBeforeSunriseR] = max(tBeforeSunriseR);
    iSunriseR(iday) = sBeforeSunriseR(ifBeforeSunriseR,1);


    %[iSunriseL:iSunriseR]

    SUNRISE_DATA = BRIGHTNESS(iSunriseL(iday):iSunriseR(iday)+1);
    SUNRISE_DATA = [SUNRISE_DATA(1) (SUNRISE_DATA(1:end-2) + SUNRISE_DATA(3:end))/2  SUNRISE_DATA(end)]; %Moving average
    %sm = 0.25; %scale
    %SUNRISE_DATA = [SUNRISE_DATA(1) (sm/2*SUNRISE_DATA(1:end-2) + (1 - sm)*SUNRISE_DATA(2:end-1) + sm/2*SUNRISE_DATA(3:end))  SUNRISE_DATA(end)]; %FIR smoother
    [dSunriseData iSunriseData]=min(SUNRISE_DATA);
    %TEST_CYCLE_TIME = TIME(CYCLE_LIMIT(1,TEST_FIRE_CYCLE_NUMBER): CYCLE_LIMIT(2,TEST_FIRE_CYCLE_NUMBER))- (TEST_FIRE_CYCLE_NUMBER-1)*24;


    iSunrise(iday) = iSunriseL(iday) + iSunriseData(end) - 1;

end


%Predict sunrise
iSunrise =   predictedSunrise(TIME, DATE, iSunrise, longitude, latitude);

%Case of missing data simulation (ACTIVATE_MISSINGSAMPLE=1), case of no
%simulation of missing data (ACTIVATE_MISSINGSAMPLE=0). In a case of
%missing data simulation, specify which gap (variable name: GAP) to simulate between 1, 2 or 3
if ACTIVATE_MISSINGSAMPLE ==1
    MISSING_LOOK_UP_TABLE = 0;
    [MISSINGDATA_FLAG_GAP1 MISSINGDATA_FLAG_GAP2 MISSINGDATA_FLAG_GAP3 START_CLOUD_GAP]= missingsamples(current_cycle,DATE,TIME,latitude, longitude,MISSING_LOOK_UP_TABLE,INDEX_ROW,INDEX_COLUMN,GAP);
    switch GAP
        case 1
            MISSINGDATA_FLAG = MISSINGDATA_FLAG_GAP1;
        case 2
            MISSINGDATA_FLAG = MISSINGDATA_FLAG_GAP2;
        case 3
            MISSINGDATA_FLAG = MISSINGDATA_FLAG_GAP3;
        otherwise
            display('Missing gap: Specify the right gap from 1 to 3');
            error('Specify the gap');
    end
end




% % % % % January = 31;
% % % % % February = 28;
% % % % % March = 31;
% % % % % April = 30;
% % % % % May = 31;
% % % % % June = 30;
% % % % %
% % % % % LAST = January+February+March+April+May+June+30;
% % % % %
% % % % %
% % % % %
% % % % %
% % % % % %N=LAST+1:LAST+15;
% % % % % N=LAST+1:LAST+23;
% % % % % B=360*(N-81)/364;
% % % % % E=(9.87*sind(2*B)-7.83*cosd(B)-1.5*sind(B))/60; % EQUATION OF TIME
% % % % % % plot(N,E)
% % % % % % grid


% TIME = 3534212172:15:3534212172-15 + 15*(NUMBER_OF_CYCLE+1)*96 ;
% TIME = TIME.';

% TIME = a(:,1);
% BRIGHTNESS = a(:,2);

% % % %
% % % % LIMITL = [];
% % % % LIMITH = [];
% % % % AVERAGE = 4.50;
% % % % %LIMITL = 1;
% % % % %LIMITL = find(TIME==TIME(1)+15*4*round(E(1))-90)
% % % % RLL = find(TIME==TIME(1)+15*4*AVERAGE*(1-round(E(1))));
% % % % LIMITL = RLL(1); %CHECK
% % % % R = TIME(LIMITL)+15*(96-96*round(E(1))-1); %CHECK R
% % % % RSS = find(TIME==R);
% % % % RS = RSS(1); %CHECK
% % % % LIMITH(1) = RSS(1); %CHECK
% % % %
% % % %
% % % %
% % % %
% % % % for i=2:NUMBER_OF_CYCLE,
% % % %     %Q = TIME(1)+15*16*round(E(i))*i;
% % % %     Q = TIME(LIMITH(i-1))+15;
% % % %     %R = TIME(RS)+15*16*round(E(i))*i;
% % % %     R = Q + 15*(96-96*round(E(i))-1);
% % % %     L = find(TIME==Q);
% % % %     H = find(TIME==R);
% % % %     LIMITL = [LIMITL L(1)]; %CHECK
% % % %     LIMITH = [LIMITH H(1)]; %CHECK
% % % % end
% % % %
% % % % LIMIT = [LIMITL;LIMITH];

% % % for i = 1:size(LIMIT,2)
% % %     bA(1:96,i) = BRIGHTNESS(LIMIT(1,i):OVERLAP:LIMIT(2,i));
% % % end

CYCLE_LIMIT = [];
for iday = 1:size(DATE,1)-1
    LIMIT = [iSunrise(iday)+1;iSunrise(iday+1)];
    CYCLE_LIMIT = [CYCLE_LIMIT LIMIT];
    dayNightLength(iday) = TIME(iSunrise(iday + 1)) - TIME(iSunrise(iday) + 1) - dayDaylength(iday);
end


% figure
% hold on
% AVERAGE = 6;
% addpath('C:\Users\JOSINE\WORK1\work\FireDetection2\models')
SAVE_MOD = [];
SAVE_To = [];
SAVE_Ta = [];
SAVE_tm = [];
SAVE_ts = [];
SAVE_w = [];
SAVE_w2 = [];
SAVE_k = [];
SAVE_dT = [];

SAVE_TRAINING_CYCLE = [];
SAVE_MEASUREMENT_ERROR = [];
number_training_cycle = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Case of a single cycle- Here removed
%TRAINING_CYCLE_NUMBER = current_cycle-1;
%TRAINING_CYCLE_NUMBER = current_cycle; %This needs to be changed in the prediction code (CHAPTER 4)
%for TRAINING_CYCLE_NUMBER = current_cycle-7:current_cycle-1;
%cycle_out_scope = 0;



TRAINING_CYCLE_NUMBER = current_cycle-1;
%for TRAINING_CYCLE_NUMBER = current_cycle-7:current_cycle-1;

cycle_out_scope = 0;

%Train using 7 previous clean cycles
NUMBER_TRAINING_CYCLE = 12;
while(number_training_cycle<NUMBER_TRAINING_CYCLE)

    iday = TRAINING_CYCLE_NUMBER;

    %     for iday = 1:size(DATE,1)-1
    cycle = BRIGHTNESS(iSunrise(iday)+1:iSunrise(iday+1));
    cycle_nomissing = cycle;
    ctime = TIME(iSunrise(iday)+1:iSunrise(iday+1));

    if ACTIVATE_MISSINGSAMPLE ==1
        missingdataFlag = MISSINGDATA_FLAG(iSunrise(iday)+1:iSunrise(iday+1));
        cycle(missingdataFlag==1) = nan;
    end


    TRAINING_CYCLE = cycle.';

    if length(cycle)<length_current_cycle
        TRAINING_CYCLE(length(cycle):length_current_cycle) = cycle(length(cycle));
    else
        TRAINING_CYCLE(length_current_cycle+1:end) = [];
    end

    TRAINING_CYCLE_TIME = ctime - (iday-1)*24;


    if length(ctime)<length_current_cycle
        %TRAINING_CYCLE_TIME(length(cycle)+1:length_current_cycle) = [TRAINING_CYCLE_TIME(length(cycle))+15/60:15/60:TRAINING_CYCLE_TIME(length(cycle))+(length_current_cycle-length(cycle))*15/60];
        TRAINING_CYCLE_TIME(length(cycle)+1:length_current_cycle) = TRAINING_CYCLE_TIME(length(cycle))+15/60:15/60:TRAINING_CYCLE_TIME(length(cycle))+(length_current_cycle-length(cycle))*15/60;
    else
        TRAINING_CYCLE_TIME(length_current_cycle+1:end) = [];
    end


    INIT_ts = daySunset(iday); %+2 hours taken as heating delay
    INIT_tm = daySolarnoon(iday)+1; %+2 hours taken as heating delay %For Duan2013 use +2 can be better.
    LENGTHOFDAY = dayDaylength(iday);
    LENGTHOFNIGHT = dayNightLength(iday);

    horizon = daySunrise90(iday); %For Inamdar_Duan2013
    tSunriseDay = TIME(iSunrise(iday))- (iday-1)*24;

    %%CHI-SQUARE FITTING
    %%
    SAMPLE_STD = [];
    %     CYCLE_CHI = 7;
    % KEEP_TRAINING_CYCLE_STD = zeros(CYCLE_CHI,length_current_cycle);
    % while(number_training_cycle<CYCLE_CHI)
    % %while(number_training_cycle<5)
    %
    %     iday = TRAINING_CYCLE_NUMBER-1; %If the top is changed -1 must be removed
    %
    %     %     for iday = 1:size(DATE,1)-1
    %     cycle = BRIGHTNESS(iSunrise(iday):iSunrise(iday+1)-1); %Change this time
    %     ctime = TIME(iSunrise(iday):iSunrise(iday+1)-1);
    %
    %
    %     TRAINING_CYCLE_STD = cycle.';
    %     if length(cycle)<length_current_cycle
    %         TRAINING_CYCLE_STD(length(cycle):length_current_cycle) = cycle(length(cycle));
    %     end
    %     TRAINING_CYCLE_TIME_STD = ctime - (iday-1)*24;
    %     if length(ctime)<length_current_cycle
    %         %TRAINING_CYCLE_TIME(length(cycle)+1:length_current_cycle) = [TRAINING_CYCLE_TIME(length(cycle))+15/60:15/60:TRAINING_CYCLE_TIME(length(cycle))+(length_current_cycle-length(cycle))*15/60];
    %         TRAINING_CYCLE_TIME_STD(length(cycle)+1:length_current_cycle) = TRAINING_CYCLE_TIME_STD(length(cycle))+15/60:15/60:TRAINING_CYCLE_TIME_STD(length(cycle))+(length_current_cycle-length(cycle))*15/60;
    %     end
    %
    %
    %
    %     number_training_cycle = number_training_cycle + 1;
    %     KEEP_TRAINING_CYCLE_STD(number_training_cycle,:) = TRAINING_CYCLE_STD(1:length_current_cycle);
    %     TRAINING_CYCLE_NUMBER = TRAINING_CYCLE_NUMBER - 1;
    %
    % end
    %
    % SAMPLE_STD = std(KEEP_TRAINING_CYCLE_STD,0,1);
    %%
    %%



    %[MOD To Ta w tm ts k dT]= diurnal_g(TRAINING_CYCLE,LENGTHOFDAY,latitude,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME); %Gottsche_2001

    %\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\MODELS
    %\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    switch MODEL
        case 'p',
            % %Parton_1981
            % %[MOD a b]= diurnal_p_Chi(TRAINING_CYCLE,LENGTHOFDAY,TRAINING_CYCLE_TIME,INIT_ts,SAMPLE_STD,LENGTHOFNIGHT); %Gottsche_2001
            %[MOD TN Ta a_P tsr_P ts Y_P Z_P ITERATIONS_A FCTCOUNT_A]= diurnal_p_Chi_modified(TRAINING_CYCLE,LENGTHOFDAY,TRAINING_CYCLE_TIME,INIT_ts,SAMPLE_STD,LENGTHOFNIGHT,daySunriseL(iday),ErrFct); %Gottsche_2001
            [MOD TN Ta a_P tsr_P ts Y_P Z_P ITERATIONS_A FCTCOUNT_A]= diurnal_p_Chi_modified(TRAINING_CYCLE,LENGTHOFDAY,TRAINING_CYCLE_TIME,INIT_ts,SAMPLE_STD,LENGTHOFNIGHT,tSunriseDay,ErrFct); %Gottsche_2001
            %thermal sunrise (tsr) - sunrise = c in Parton_1981
            To = TN;
            tm = [];
            dT = [];
            w = [];
            k = [];
            w2 = [];
            tau = [];
            m_z = [];
            P = [];
            v_z = [];
            v_m = [];
            theta_z = [];
            theta_zm = [];
            theta_zs = [];
            m_zs = [];
            a_RKHS = [];
            K_RKHS = [];
            deltaK_RKHS = [];
            a_SVD = [];
            U_SVD = [];
            deltaU_SVD = [];
            tmin = [];

        case 's',
            %Schadlich_2001
            [MOD To Ta w tm ts k ITERATIONS_A FCTCOUNT_A]= diurnal_s_Chi(TRAINING_CYCLE,LENGTHOFDAY,latitude,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME,SAMPLE_STD,ErrFct); %Schadlich_2001
            dT = [];
            w = [];
            k = [];
            w2 = [];
            tau = [];
            m_z = [];
            P = [];
            v_z = [];
            v_m = [];
            theta_z = [];
            theta_zm = [];
            theta_zs = [];
            m_zs = [];
            tsr_P = [];
            a_P = [];
            Y_P = [];
            Z_P = [];
            a_RKHS = [];
            K_RKHS = [];
            deltaK_RKHS = [];
            a_SVD = [];
            U_SVD = [];
            deltaU_SVD = [];
            tmin = [];

        case 'g',
            % %Gottsche_2001
            % %[MOD To Ta w tm ts k dT]= diurnal_g(TRAINING_CYCLE,LENGTHOFDAY,latitude,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME); %Gottsche_2001
            [MOD To Ta w tm ts k dT ITERATIONS_A FCTCOUNT_A]= diurnal_g_Chi(TRAINING_CYCLE,LENGTHOFDAY,latitude,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME,SAMPLE_STD,ErrFct); %Gottsche_2001
            w = [];
            k = [];
            w2 = [];
            tau = [];
            m_z = [];
            P = [];
            v_z = [];
            v_m = [];
            theta_z = [];
            theta_zm = [];
            theta_zs = [];
            m_zs = [];
            tsr_P = [];
            a_P = [];
            Y_P = [];
            Z_P = [];
            a_RKHS = [];
            K_RKHS = [];
            deltaK_RKHS = [];
            a_SVD = [];
            U_SVD = [];
            deltaU_SVD = [];
            tmin = [];

        case 'j',
            %Jiang_2006
            [MOD a b td ts beta alpha ITERATIONS_A FCTCOUNT_A] = diurnal_j_Chi(TRAINING_CYCLE,LENGTHOFDAY,latitude,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME,SAMPLE_STD,ErrFct); %Schadlich_2001
            k = -1/alpha;
            To = a;
            Ta = b;
            w = pi/beta;
            tau = [];
            tm = td;
            dT = [];
            w2 = [];
            tau = [];
            m_z = [];
            P = [];
            v_z = [];
            v_m = [];
            theta_z = [];
            theta_zm = [];
            theta_zs = [];
            m_zs = [];
            tsr_P = [];
            a_P = [];
            Y_P = [];
            Z_P = [];
            a_RKHS = [];
            K_RKHS = [];
            deltaK_RKHS = [];
            a_SVD = [];
            U_SVD = [];
            deltaU_SVD = [];
            tmin = [];


        case 'v',
            %vandenBergh_2007
            [MOD To Ta w1 w2 tm ts k  ITERATIONS_A FCTCOUNT_A] = diurnal_v_Chi(TRAINING_CYCLE,LENGTHOFDAY,latitude,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME,SAMPLE_STD,ErrFct); %van den Bergh_2006
            w = w1;
            tau = [];
            dT = [];
            kTemp = k;
            %k = [];
            m_z = [];
            P = [];
            v_z = [];
            v_m = [];
            theta_z = [];
            theta_zm = [];
            theta_zs = [];
            m_zs = [];
            tsr_P = [];
            a_P = [];
            Y_P = [];
            Z_P = [];
            a_RKHS = [];
            K_RKHS = [];
            deltaK_RKHS = [];
            a_SVD = [];
            U_SVD = [];
            deltaU_SVD = [];
            tmin = [];
            TTTMMM = TRAINING_CYCLE_TIME;
        case 'i',
            %Inamdar_2008
            [MOD To Ta w tm ts k dT ITERATIONS_A FCTCOUNT_A]= diurnal_i_Chi(TRAINING_CYCLE,LENGTHOFDAY,latitude,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME,SAMPLE_STD,ErrFct); %Gottsche_2001
            w = [];
            k = [];
            w2 = [];
            tau = [];
            m_z = [];
            P = [];
            v_z = [];
            v_m = [];
            theta_z = [];
            theta_zm = [];
            theta_zs = [];
            m_zs = [];
            tsr_P = [];
            a_P = [];
            Y_P = [];
            Z_P = [];
            a_RKHS = [];
            K_RKHS = [];
            deltaK_RKHS = [];
            a_SVD = [];
            U_SVD = [];
            deltaU_SVD = [];
            tmin = [];

        case 'd',
            %Modification of Inamdar_2008 by Duan2013
            %[MOD To Ta w tm ts k dT ITERATIONS_A FCTCOUNT_A]= diurnal_iD_Chi(TRAINING_CYCLE,daySunriseL(iday),latitude,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME,SAMPLE_STD,ErrFct); %Gottsche_2001
            %[MOD To Ta w tm ts k dT ITERATIONS_A FCTCOUNT_A]= diurnal_iD_Chi(TRAINING_CYCLE,tSunriseDay,latitude,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME,SAMPLE_STD,ErrFct); %Gottsche_2001
            [MOD To Ta w tm ts k dT ITERATIONS_A FCTCOUNT_A]= diurnal_iD_Chi(TRAINING_CYCLE,horizon,latitude,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME,SAMPLE_STD,ErrFct); %Gottsche_2001
            k = [];
            w2 = [];
            tau = [];
            m_z = [];
            P = [];
            v_z = [];
            v_m = [];
            theta_z = [];
            theta_zm = [];
            theta_zs = [];
            m_zs = [];
            tsr_P = [];
            a_P = [];
            Y_P = [];
            Z_P = [];
            a_RKHS = [];
            K_RKHS = [];
            deltaK_RKHS = [];
            a_SVD = [];
            U_SVD = [];
            deltaU_SVD = [];
            tmin = [];


            %[MOD To Ta w tm ts k dT]= diurnal_iD_Chi(TRAINING_CYCLE,TIME(iSunrise(iday)) - (iday - 1)*24,latitude,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME,SAMPLE_STD); %Gottsche_2001

            % for iday = 1:size(DATE,1)
            %     TIME(iSunrise(iday)) - (iday - 1)*24
            % end

        case 'z',
            %Gottsche_2009
            [MOD To Ta tm ts k dT tau m_z P v_z v_m theta_z theta_zm theta_zs m_zs ITERATIONS_A FCTCOUNT_A] = diurnal_gz_Chi(TRAINING_CYCLE,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME,DATE(iday,1),DATE(iday,2),DATE(iday,3),longitude,latitude,SAMPLE_STD,ErrFct); %Gottsche_2001
            w = [];
            %%%%%%%%%%%k = []; %k is not cleared
            w2 = [];
            tsr_P = [];
            a_P = [];
            Y_P = [];
            Z_P = [];
            a_RKHS = [];
            K_RKHS = [];
            deltaK_RKHS = [];
            a_SVD = [];
            U_SVD = [];
            deltaU_SVD = [];
            tmin = [];

        case 'r',
            %RKHS Minimum-Norm Least-Squares (vandenBergh_2007)
            [MOD a_RKHS K_RKHS deltaK_RKHS ITERATIONS_A FCTCOUNT_A]= diurnal_rkhs_Chi(TRAINING_CYCLE.',TRAINING_CYCLE_TIME,SAMPLE_STD,ErrFct);
            %RKHS lagrange relaxation  (Udahemuka_2008)
            %[MOD a]= diurnal_rkhs_Chi(TRAINING_CYCLE.',TRAINING_CYCLE_TIME,SAMPLE_STD,'lr');
            %RKHS regularization.
            %[MOD a]= diurnal_rkhs_Chi(TRAINING_CYCLE.',TRAINING_CYCLE_TIME,SAMPLE_STD,'reg');
            To = [];
            tm = [];
            dT = [];
            w = [];
            k = [];
            w2 = [];
            tau = [];
            m_z = [];
            P = [];
            v_z = [];
            v_m = [];
            theta_z = [];
            theta_zm = [];
            theta_zs = [];
            m_zs = [];
            tsr_P = [];
            a_P = [];
            Y_P = [];
            Z_P = [];
            Ta = [];
            tm = [];
            ts = [];
            a_SVD = [];
            U_SVD = [];
            deltaU_SVD = [];
            tmin = [];

        case 't',
            %SVD Minimum-Norm Least-Squares
            [MOD a_SVD U_SVD deltaU_SVD ITERATIONS_A FCTCOUNT_A]= diurnal_svd_Chi(TRAINING_CYCLE.',TRAINING_CYCLE_TIME,SAMPLE_STD,KEEP_TRAINING_CYCLE_STD,ErrFct);
            To = [];
            tm = [];
            dT = [];
            w = [];
            k = [];
            w2 = [];
            tau = [];
            m_z = [];
            P = [];
            v_z = [];
            v_m = [];
            theta_z = [];
            theta_zm = [];
            theta_zs = [];
            m_zs = [];
            tsr_P = [];
            a_P = [];
            Y_P = [];
            Z_P = [];
            Ta = [];
            ts = [];
            a_RKHS = [];
            K_RKHS = [];
            deltaK_RKHS = [];
            tmin = [];

            %SVD robust (Udahemuka_2008)
            %[MOD a]=
            %diurnal_svd_Chi(TRAINING_CYCLE.',TRAINING_CYCLE_TIME,SAMPLE_STD,KEEP_TRAINING_CYCLE_STD,'rob');

        case 'u',
            %DDTP (Duan_2013)
            %TRAINING_CYCLE_TIME over all numberCycle

            numberCycle_DDTP = DDTP_Nbre_Cycles;
            iday = current_cycle;
            cycle_DDTP = BRIGHTNESS(iSunrise(iday)+1:iSunrise(iday+DDTP_Nbre_Cycles));
            ctime_DDTP = TIME(iSunrise(iday)+1:iSunrise(iday+DDTP_Nbre_Cycles));

            TRAINING_CYCLE_DDTP = cycle_DDTP.';

            %TRAINING_CYCLE_TIME_DDTP = ctime - (iday-1)*24;
            TRAINING_CYCLE_TIME_DDTP = ctime_DDTP; %- (iday-1)*24;


            INIT_ts_DDTP = sunset(iday:iday+DDTP_Nbre_Cycles-1);         %daySunset(iday:iday+3); %+2 hours taken as heating delay
            INIT_tm_DDTP = solarnoon(iday:iday+DDTP_Nbre_Cycles)+1;    %daySolarnoon(iday:iday+4)+1; %+2 hours taken as heating delay
            %LENGTHOFDAY_DDTP = dayDaylength(iday);
            %LENGTHOFNIGHT_DDTP = dayNightLength(iday);

            samplePerCycle = (CYCLE_LIMIT(2,:) - CYCLE_LIMIT(1,:)) + 1;
            %sssss=size(samplePerCycle)
            %TIME(iSunrise(iday)) - (iday - 1)*24 _DDTP
            %tmin_DDTP = TIME(iSunrise(iday)) - (iday - 1)*24


            %SAMPLE_STD_DDTP = std(KEEP_TRAINING_CYCLE_STD,0,1); %Not yet correct, To be checked for correctness
            SAMPLE_STD_DDTP = [];
            for stdi = iday:iday+DDTP_Nbre_Cycles-1
                %TEMP_SAMPLE_STD = [SAMPLE_STD SAMPLE_STD(end)*ones(1,samplePerCycle(stdi)-length(SAMPLE_STD))];
                if samplePerCycle(stdi) > length(SAMPLE_STD)
                    TEMP_SAMPLE_STD = [SAMPLE_STD SAMPLE_STD(end)*ones(1,samplePerCycle(stdi)-length(SAMPLE_STD))];
                elseif samplePerCycle(stdi) < length(SAMPLE_STD)
                    TEMP_SAMPLE_STD = SAMPLE_STD(1:samplePerCycle(stdi));
                else
                    TEMP_SAMPLE_STD = SAMPLE_STD;
                end
                SAMPLE_STD_DDTP = [SAMPLE_STD_DDTP TEMP_SAMPLE_STD];
            end


            %iday
            %Given the start of cycle day: iday
            %[MOD To Ta w tm ts k dT]= diurnal_d_Chi(TRAINING_CYCLE_DDTP,TIME(iSunrise(iday:iday+4)) - (iday - 1)*24,latitude,INIT_tm_DDTP,INIT_ts_DDTP,TRAINING_CYCLE_TIME_DDTP,SAMPLE_STD_DDTP,numberCycle_DDTP,TIME(iSunrise(iday:iday+4)) - (iday - 1)*24, samplePerCycle(iday:iday+3));
            [MOD To Ta w tm ts k dT tmin ITERATIONS_A FCTCOUNT_A]= diurnal_d_Chi(TRAINING_CYCLE_DDTP,TIME(iSunrise(iday:iday+DDTP_Nbre_Cycles)),latitude,INIT_tm_DDTP,INIT_ts_DDTP,TRAINING_CYCLE_TIME_DDTP,SAMPLE_STD_DDTP,numberCycle_DDTP,TIME(iSunrise(iday:iday+DDTP_Nbre_Cycles)), samplePerCycle(iday:iday+DDTP_Nbre_Cycles-1),ErrFct); %tmin=84, ttsr=84
            %[MOD To Ta w tm ts k dT tmin ITERATIONS_A FCTCOUNT_A]= diurnal_d_Chi(TRAINING_CYCLE_DDTP,morningHorizonTime(iday:iday+DDTP_Nbre_Cycles),latitude,INIT_tm_DDTP,INIT_ts_DDTP,TRAINING_CYCLE_TIME_DDTP,SAMPLE_STD_DDTP,numberCycle_DDTP,TIME(iSunrise(iday:iday+DDTP_Nbre_Cycles)), samplePerCycle(iday:iday+DDTP_Nbre_Cycles-1),ErrFct); %tmin=84, ttsr=90
            %[MOD To Ta w tm ts k dT tmin ITERATIONS_A FCTCOUNT_A]= diurnal_d_Chi(TRAINING_CYCLE_DDTP,morningHorizonTime(iday:iday+DDTP_Nbre_Cycles),latitude,INIT_tm_DDTP,INIT_ts_DDTP,TRAINING_CYCLE_TIME_DDTP,SAMPLE_STD_DDTP,numberCycle_DDTP,morningHorizonTime(iday:iday+DDTP_Nbre_Cycles), samplePerCycle(iday:iday+DDTP_Nbre_Cycles-1),ErrFct); %tmin=90, ttsr=90
            %Choice of 84 or 90 as ttsr to compare against.

            %w = [];
            %k = [];
            w2 = [];
            tau = [];
            m_z = [];
            P = [];
            v_z = [];
            v_m = [];
            theta_z = [];
            theta_zm = [];
            theta_zs = [];
            m_zs = [];
            tsr_P = [];
            a_P = [];
            Y_P = [];
            Z_P = [];
            a_RKHS = [];
            K_RKHS = [];
            deltaK_RKHS = [];
            a_SVD = [];
            U_SVD = [];
            deltaU_SVD = [];


        otherwise
            disp('Choose model: Specify the right model');
            error('Specify the right model');
    end




    %\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    %\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    %     if ((tm < 9) | (tm>14)) | (tm > ts) | (Ta <=0) | ((ts <13) | (ts > 20)) | (k>100) | (w <0) %Use also reduced chi-square to select cycles
    %     if ((tm < 10) | (tm>14)) | (tm > ts) | (Ta <=0) | ((ts <13) | (ts > 20)) | (k>100) | (w <0) %Use also reduced chi-square to select cycles

    %(k<0.05)&(k>100)): both give a flat response after time ts
    %Constraint on the DTC to be accepted as a model
%     (tm < daySolarnoon(iday)) 
%     (tm>daySolarnoon(iday)+6))
%     (tm > ts) 
%     (Ta <=0) 
%     ((ts <daySunset(iday)-3) 
%     (ts > daySunset(iday)+3)) 
%     ((k<0.05) || (k>100)) 
%     (w <0) %Use also reduced chi-square to select cycles
    
    %if ((tm < daySolarnoon(iday)) || (tm>daySolarnoon(iday)+6)) || (tm > ts) || (Ta <=0) || ((ts <daySunset(iday)-3) || (ts > daySunset(iday)+3)) || ((kTemp<0.05) || (kTemp>100)) || (w <0) %Use also reduced chi-square to select cycles
    if ((tm < daySolarnoon(iday)) || (tm>daySolarnoon(iday)+6)) || (tm > ts - 1) || (Ta <=0) || ((ts <daySunset(iday)-3) || (ts > daySunset(iday)+3)) || (kTemp>100) || (w <0) %Use also reduced chi-square to select cycles
        current_cycle_select = 0;
        %Display them to check why a given cycle was removed from training model
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        %tm                      %
        %ts                      %
        %daySolarnoon(iday)      %
        %daySunset(iday)         %
        %k                       %
        %%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        current_cycle_select = 1;
    end
    kTemp = [];

    if current_cycle_select ==1
        SAVE_MOD = [SAVE_MOD MOD.'];
        SAVE_TRAINING_CYCLE =  [SAVE_TRAINING_CYCLE TRAINING_CYCLE];
        SAVE_MEASUREMENT_ERROR = [SAVE_MEASUREMENT_ERROR (TRAINING_CYCLE-MOD.')];
        number_training_cycle = number_training_cycle + 1;
        SAVE_To =[SAVE_To To];
        SAVE_Ta =[SAVE_Ta Ta];
        SAVE_tm =[SAVE_tm tm];
        SAVE_ts =[SAVE_ts ts];
        SAVE_w = [SAVE_w w]; %MUST BE REMOVED
        SAVE_w2 = [SAVE_w2 w2]; %MUST BE REMOVED
        SAVE_k =[SAVE_k k];
        %SAVE_dT =[SAVE_dT dT];
    else %Case a cycle is not selected as a model replace it with NaN
        SAVE_MOD = [SAVE_MOD nan(size(MOD.'))];
        SAVE_TRAINING_CYCLE =  [SAVE_TRAINING_CYCLE nan(size(TRAINING_CYCLE))];
        SAVE_MEASUREMENT_ERROR = [SAVE_MEASUREMENT_ERROR nan(size(TRAINING_CYCLE-MOD.'))];
        %number_training_cycle = number_training_cycle + 1;
        SAVE_To =[SAVE_To NaN];
        SAVE_Ta =[SAVE_Ta NaN];
        SAVE_tm =[SAVE_tm NaN];
        SAVE_ts =[SAVE_ts NaN];
        SAVE_w = [SAVE_w NaN]; %MUST BE REMOVED
        SAVE_w2 = [SAVE_w2 NaN]; %MUST BE REMOVED
        SAVE_k =[SAVE_k NaN];
        %SAVE_dT =[SAVE_dT NaN];
    end



    if number_training_cycle<NUMBER_TRAINING_CYCLE %7
        TRAINING_CYCLE_NUMBER = TRAINING_CYCLE_NUMBER - 1;
    end

    if TRAINING_CYCLE_NUMBER<=0 %Case no 7 previous cycles to use in training
        %disp('FINISH WITH AVAILABLE CYCLES NO DECISION CAN BE TAKEN CONCERNING THE CYCLE UNDER TEST');
        cycle_out_scope = 1;
        break;
    end
    %         [MOD a b td ts beta alpha]= diurnal_g(TRAINING_CYCLE,LENGTHOFDAY,latitude,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME);
    %Train with more cycles and use chi-square fitting and other
    %approaches such as MSE and so on.
    %Call also robust fitting and RKHS models and Göttsche_2009 and Frans model and Göttsche_1999

    %kcycle = [kcycle cycle];
    %kctime = [kctime ctime];

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     figure
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %length(cycle)
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     plot(ctime,cycle,'r*')
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     hold on
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     plot(ctime,MOD)
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     hold off
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     legend('Valid observed samples','Interpolated curve')
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     xlabel('Time stamp')
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     ylabel('Brightness temperature (Kelvin)')
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     title('Fitting from the training')
    %title('TRAINING')
    %DTR(iday) = max(cycle) - min(cycle);
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%length(TRAINING_CYCLE)

% % % % % % % % % % % % % % % % if length(cycle)<length_current_cycle
% % % % % % % % % % % % % % % %     TRAINING_CYCLE(length(cycle):length_current_cycle) = cycle(length(cycle));
% % % % % % % % % % % % % % % % end


% % % % % % % % % % % % % % % % if length(ctime)<length_current_cycle
% % % % % % % % % % % % % % % %     %TRAINING_CYCLE_TIME(length(cycle)+1:length_current_cycle) = [TRAINING_CYCLE_TIME(length(cycle))+15/60:15/60:TRAINING_CYCLE_TIME(length(cycle))+(length_current_cycle-length(cycle))*15/60];
% % % % % % % % % % % % % % % %     TRAINING_CYCLE_TIME(length(cycle)+1:length_current_cycle) = TRAINING_CYCLE_TIME(length(cycle))+15/60:15/60:TRAINING_CYCLE_TIME(length(cycle))+(length_current_cycle-length(cycle))*15/60;
% % % % % % % % % % % % % % % % end

%length(TRAINING_CYCLE)

%%%%%
%%%%%
%iday = current_cycle; %[This should be included to get no delayed time]


%%%%
%%%%










%SVD Minimum-Norm Least-Squares
% [MOD a_SVD U_SVD deltaU_SVD]= diurnal_svd_Chi(TRAINING_CYCLE.',TRAINING_CYCLE_TIME,SAMPLE_STD,KEEP_TRAINING_CYCLE_STD,'mnls');
% To = [];
% tm = [];
% dT = [];
% w = [];
% k = [];
% w2 = [];
% tau = [];
% m_z = [];
% P = [];
% v_z = [];
% v_m = [];
% theta_z = [];
% theta_zm = [];
% theta_zs = [];
% m_zs = [];
% tsr_P = [];
% a_P = [];
% Y_P = [];
% Z_P = [];
% Ta = [];
% tm = [];
% ts = [];
% a_RKHS = [];
% K_RKHS = [];
% deltaK_RKHS = [];


%SVD robust (Udahemuka_2008)
%[MOD a]= diurnal_svd_Chi(TRAINING_CYCLE.',TRAINING_CYCLE_TIME,SAMPLE_STD,KEEP_TRAINING_CYCLE_STD,'rob');



%function [c1 To Ta w tm ts k dT]= diurnal_g(b,tsr,phi,tm,ts,TIME,SAMPLE_STD)


INITTs = INIT_ts;
%RETTs = ts
INITTm = INIT_tm;
%RETTm = tm

%length(MOD)

% size(TRAINING_CYCLE)
% size(MOD)
%plot(TEST_CYCLE_TIME, MOD,'r')


%Results (RMSE, MAE and BIAS) over only missing data gaps
RESIDUAL_OVER_MISSING_VALUE_ONLY = [];
RMSE_NSQ_OVER_MISSING_VALUE_ONLY = [];
MAE_NSQ_OVER_MISSING_VALUE_ONLY = [];
BIAS_NSQ_OVER_MISSING_VALUE_ONLY = [];
if ACTIVATE_MISSINGSAMPLE ==1
    RESIDUAL_OVER_MISSING_VALUE_ONLY = MOD(missingdataFlag==1) - cycle_nomissing(missingdataFlag==1);
    RMSE_NSQ_OVER_MISSING_VALUE_ONLY = sqrt(mean(RESIDUAL_OVER_MISSING_VALUE_ONLY.^2));
    MAE_NSQ_OVER_MISSING_VALUE_ONLY = mean(abs(RESIDUAL_OVER_MISSING_VALUE_ONLY));
    BIAS_NSQ_OVER_MISSING_VALUE_ONLY = sum(RESIDUAL_OVER_MISSING_VALUE_ONLY);

    %ROMVO = RESIDUAL_OVER_MISSING_VALUE_ONLY;
    %RMSEOMVO = RMSE_NSQ_OVER_MISSING_VALUE_ONLY
    %MAEOMVO = MAE_NSQ_OVER_MISSING_VALUE_ONLY
    %BIASOMVO = BIAS_NSQ_OVER_MISSING_VALUE_ONLY
end


%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
%Train using 7 previous clean cycles


%To_M = To;
%Ta_M = Ta;
%tm_M = tm;
%ts_M = ts;
dT_M = dT;
%w_M = w;
%k_M = k;
%w2_M = w2;
tau_M = tau;
m_z_M = m_z;
P_M = P;
v_z_M = v_z;
v_m_M = v_m;
theta_z_M = theta_z;
theta_zm_M = theta_zm;
theta_zs_M = theta_zs;
m_zs_M = m_zs;
tsr_P_M = tsr_P;
a_P_M = a_P;
Y_P_M = Y_P;
Z_P_M = Z_P;
a_RKHS_M = a_RKHS;
K_RKHS_M = K_RKHS;
deltaK_RKHS_M = deltaK_RKHS;
a_SVD_M = a_SVD;
U_SVD_M = U_SVD;
deltaU_SVD_M = deltaU_SVD;
tmin_M = tmin;
%Qc = 1;
%Rc = 1;
%SAVE_TRAINING_CYCLE = 1;
%SAVE_NOISE_MEAS = 1;
%mp = 1;
%cycle_out_scope = 1;
%[length(MOD) length(TRAINING_CYCLE)]


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

%7 DAY'S MODEL MEAN AND STANDARD DEVIATION
mp = nanmean(SAVE_MOD,2);
vpn = nanstd(SAVE_MOD,'',2); %Unbiased estimate

%plot(mp)

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hist(SAVE_MOD(1,:)-mp(1)); %START OF CYCLE HISTOGRAM FOR PROCESS NOISE
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % title('ONE CYCLE (1) - MEAN')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hist(SAVE_MOD(48,:)-mp(48)); %ANOTHER CYCLE TIME HISTOGRAM FOR PROCESS NOISE
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % title('ANOTHER CYCLE (48) - MEAN')


%FINDING MEAN OF EACH PARAMETER OVER 7 DAYS, IN ORDER TO HAVE A SINGLE
%MODEL AT CURRENT DAY
To_M = nanmean(SAVE_To);
Ta_M = nanmean(SAVE_Ta);
tm_M = nanmean(SAVE_tm);
ts_M = nanmean(SAVE_ts);
w_M = nanmean(SAVE_w);   %CONSIDER IT THE TEST THE MODEL
w2_M = nanmean(SAVE_w2);   %CONSIDER IT THE TEST THE MODEL
%dT_M = nanmean(SAVE_dT);
%t = TRAINING_CYCLE_TIME; %CONSIDER IT THE TEST THE MODEL
%k = w/pi*((1/tan(pi/w*(ts-tm)))-dT/Ta*(1/sin(pi/w*(ts-tm))));
k_M = nanmean(SAVE_k);   %CONSIDER IT THE TEST THE MODEL


fParameter = [SAVE_To
SAVE_Ta
SAVE_tm
SAVE_ts
SAVE_w
SAVE_w2];
%fParameter

% MParameter = nanmean(fParameter,2);
% CovParameter = 0;
% for i = 1:size(fParameter,2)  
%     if sum(isnan(fParameter(:,i)))==0
%         CovParameter = CovParameter + (fParameter(:,i)-MParameter)*(fParameter(:,i)-MParameter).';
%     end
% end
% NTRAININGCYCLE = sum(~isnan(fParameter(1,:)));
% CovParameter = CovParameter/(NTRAININGCYCLE-1); %(size(fParameter,2)-1);

%THIS MODEL CAN BE USED TO CHECK IF c1 = mp
%c1 = (To_M+Ta_M*cos(pi/w_M*(t-tm_M))).*(t<ts_M)+((To_M+dT_M)+(Ta_M*cos(pi/w*(ts_M-tm_M))-dT_M)*exp(-(t-ts_M)/k_M)).*(t>=ts_M);
%THE RIGHT MODEL IS THE BOTTOM ONE, NOT THE TOP ONE
% c1_1 = (To_M+Ta_M*cos(pi/w_M*(t-tm_M)));%.*(t<ts_M)
% c1_1(t>=ts_M)=0;
% c1_2 = ((To_M+dT_M)+(Ta_M*cos(pi/w*(ts_M-tm_M))-dT_M)*exp(-(t-ts_M)/k_M));%.*(t>=ts_M);
% c1_2(t<ts_M)=0;
% c1 = c1_1 + c1_2;

%length(find(isnan(c1)==1))

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % plot(c1,'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold on
%TRAIN AGAIN TO GET A MODEL OVER 7 DAYS, BY USING TRAINING THE MEAN OF THE
%7 DAY'S MODEL AND INITIALIZING WITH MEAN OF 7 DAY'S MODELS
% [MOD To Ta w tm ts k dT]= diurnal_g(mp,w,latitude,tm,ts,TRAINING_CYCLE_TIME); %Gottsche_2001
% c2 = (To+Ta*cos(pi/w*(t-tm))).*(t<ts)+((To+dT)+(Ta*cos(pi/w*(ts-tm))-dT)*exp(-(t-ts)/k)).*(t>=ts);
% plot(c2,'g')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % errorbar(mp,vpn)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % title('Model from sample mean and from mean of parameters')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % legend('Model from mean','Model from mean of parameters')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold off

Qc = vpn.^2;  %Covariance in process noise

% SAVE_NOISE_MEAS = [];
% for i = 1:NUMBER_OF_CYCLE_USED_TO_TRAIN
%     NOISE_MEAS = SAVE_TRAINING_CYCLE(:,i) - mp;%SAVE_MOD;
%     SAVE_NOISE_MEAS = [SAVE_NOISE_MEAS NOISE_MEAS];
%     %SAVE_NOISE_TRANS(:,i) = NOISE_MEAS;
% end



% ALL_CYCLE = BRIGHTNESS(LIMIT(1,1): OVERLAP:LIMIT(2,NUMBER_OF_CYCLE));
%
% size(ALL_CYCLE)
% size(mp)

% for i = 1:NUMBER_OF_CYCLE
%     NOISE_MEAS = BRIGHTNESS(LIMIT(1,i): LIMIT(2,i)) - mp;%SAVE_MOD;
%     SAVE_NOISE_MEAS = [SAVE_NOISE_MEAS NOISE_MEAS];
%     SAVE_NOISE_TRANS(:,i) = NOISE_MEAS;
% end

SAVE_NOISE_MEAS = SAVE_MEASUREMENT_ERROR;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % SAVE_NOISE_MEAS = [];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % for i = 1:NUMBER_OF_CYCLE_USED_TO_TRAIN
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     NOISE_MEAS = SAVE_TRAINING_CYCLE(:,i) - mp;%SAVE_MOD;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     SAVE_NOISE_MEAS = [SAVE_NOISE_MEAS NOISE_MEAS];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %SAVE_NOISE_TRANS(:,i) = NOISE_MEAS;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % end


% Vmn = mean(mean(SAVE_NOISE_MEAS,1),2);
% Rc = Vmn.^2;
%vmn = std(SAVE_NOISE_MEAS,'',2);
%Rc = mean(vmn.*vmn);  %Covariance in process noise

% % % for i = 1:size(LIMIT,2)
% % %     bA(1:96,i) = BRIGHTNESS(LIMIT(1,i):OVERLAP:LIMIT(2,i));
% % % end
% % % % % % % % % % % % % % % % % % % % % % % tp = 4.2:15/60:4.2+24-15/60;
% % % % % % % % % % % % % % % % % % % % % % % figure
% % % % % % % % % % % % % % % % % % % % % % % subplot(212)
% % % % % % % % % % % % % % % % % % % % % % % plot(tp,Qc)
% % % % % % % % % % % % % % % % % % % % % % % xlabel('Time stamp (Hour)')
% % % % % % % % % % % % % % % % % % % % % % % ylabel('Variance in brightness temperature')
% % % % % % % % % % % % % % % % % % % % % % % %hold on
% % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % subplot(211)
% % % % % % % % % % % % % % % % % % % % % % % xlabel('Time stamp (Hour)')
% % % % % % % % % % % % % % % % % % % % % % % ylabel('Brightness temperature (Kelvin)')
% % % % % % % % % % % % % % % % % % % % % % % plot(tp,mp)
% % % % % % % % % % % % % % % % % % % % % % % %hold off
% vmn = mean(SAVE_NOISE_MEAS,2);
vmn = reshape(SAVE_NOISE_MEAS, 1, size(SAVE_NOISE_MEAS,1)*size(SAVE_NOISE_MEAS,2));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hist(vmn);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % title('Measurement noise')

Rc = nanvar(vmn); %Unbiased estimate
% randn('state',sa);



%If at least one cycle is used to train (one with current_cycle_select =1, then there is no problem with the following values, but in case no cycle is found,
%there will return a single value NaN instead of returning multiple value NaN)
%SAVE_NOISE_MEAS cycle_out_scope



%%%%
%%%%
% figure
% plot(mp,'b')
% hold on 



To = To_M;
Ta = Ta_M;
tm = tm_M;
ts = ts_M;
w1 = w_M;   
w2 = w2_M; 
k = k_M;
t = TTTMMM;


if Ta~=0
    % k = w/pi*(atan(pi/w*(ts-tm))-dT/Ta*asin(pi/w*(ts-tm)));

    %Calculation of the exponential damping factor(initialisation)
    %k = w/pi*(atan(pi/w*(ts-tm))-dT/Ta*asin(pi/w*(ts-tm)));  %pi/w*(ts-tm) must be in [-1,1]
    %k = w2/pi*(1/tan(pi/w2*(ts-tm)));%-dT/Ta*(1/sin(pi/w*(ts-tm))));  %pi/w*(ts-tm) must be in [-1,1]

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

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % plot(c1,'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % LL = BRIGHTNESS(iSunrise(current_cycle)+1:iSunrise(current_cycle+1));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % plot(LL,'bx')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % plot(mp,'r')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold off

% LL = BRIGHTNESS(iSunrise(current_cycle)+1:iSunrise(current_cycle+1));
% 
% sqrt(mean((LL(:) - mp(:)).^2))
% mean(abs(LL(:) - mp(:)))
% 
% sqrt(mean((c1(:) - LL(:)).^2))
% mean(abs(c1(:) - LL(:)))


%[MOD To Ta w1 w2 tm ts k  ITERATIONS_A FCTCOUNT_A] = diurnal_v_Chi(mp,LENGTHOFDAY,latitude,tm_M,ts_M,TRAINING_CYCLE_TIME,SAMPLE_STD,ErrFct); %van den Bergh_2006
%[To_M Ta_M w_M w2_M tm_M ts_M k_M]
%[To Ta w1 w2 tm ts k]
%TTTNOISE = t

%daySolarnoon 
%daySunset