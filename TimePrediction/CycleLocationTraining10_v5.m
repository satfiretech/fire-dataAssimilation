function  [MOD,TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude,longitude,column, row,To_M, Ta_M, tm_M, ts_M, dT_M, k_M, w2_M, tau_M, m_z_M, P_M, v_z_M, v_m_M, theta_z_M, theta_zm_M, theta_zs_M, m_zs_M, tsr_P_M, a_P_M, Y_P_M, Z_P_M, a_RKHS_M, K_RKHS_M, deltaK_RKHS_M, a_SVD_M U_SVD_M deltaU_SVD_M Q, R1, bA, MNs, mp, cycle_out_scope,SunriseDay, SunsetDay, StartDay, tSunriseDay, tSunsetDay, tStartDay, tDayLength, tSolarnoonDay sr ITERATIONS_A FCTCOUNT_A RMSE_NSQ_OVER_MISSING_VALUE_ONLY MAE_NSQ_OVER_MISSING_VALUE_ONLY BIAS_NSQ_OVER_MISSING_VALUE_ONLY tmin_M horizon missingdataFlag tmorningHorizon vtSunriseDay]= CycleLocationTraining(COLUMN,ROW,TEST_FIRE_CYCLE_NUMBER,FNDATAREGION, MODEL,ErrFct,ACTIVATE_MISSINGSAMPLE,GAP, DDTP_Nbre_Cycles)
              

%Cycle to test, and its location and its training
%OPTIMIZATION MUST BE DONE ON NUMBER OF TIME LOCATION DATA ARE REPEATEDLY
%ACQUIRED


% addpath('C:\Users\JOSINE\WORK1\work\FireDetection2\solarTimeLocation');
% addpath('C:\Users\JOSINE\WORK1\work\FireDetection2\models');
% addpath('C:\Users\JOSINE\WORK1\work\FireDetection2\TimePrediction');


rw = ROW - 778 + 1;  %LINE
cm = COLUMN - 822 + 1; %COLUMN(PIXEL)

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     latitude = coordContent(1);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     longitude = coordContent(2);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     column = coordContent(3);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     row = coordContent(4);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         DATA = load(file);%load('msgcsv/pix_88806.txt');

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %       current_txt_file = file;
current_txt_file = FNDATAREGION(rw,cm).filename;
current_txt_file(1) = 'C';

latitude = FNDATAREGION(rw,cm).latitude;
longitude = FNDATAREGION(rw,cm).longitude;
row = ROW;
column = COLUMN;
BRIGHTNESS = FNDATAREGION(rw,cm).Brightness;
TIME = (0+12:15:(length(BRIGHTNESS)-1)*15+12)/60; %Time taken from 23-07-2007 00:00. This time is 0. The time is expressed in hours.

%Approximate sunrise, sunset, day length and solar noon for a given
%pixel for all the days taken into account
%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         DATE = [2007 7 23;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 7 24;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 7 25;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 7 26;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 7 27;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 7 28;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 7 29;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 7 30;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 7 31;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 8 1;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 8 2;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 8 3;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 8 4;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 8 5;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 8 6;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 8 7;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 8 8;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 8 9;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 8 10;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 8 11;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 8 12;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 8 13;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             2007 8 14];


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
    2007 8 5];


daySunriseL = zeros(1,size(DATE,1));
daySunriseR = zeros(1,size(DATE,1));
daySolarnoon = zeros(1,size(DATE,1));
daySunset = zeros(1,size(DATE,1));
dayDaylength = zeros(1,size(DATE,1));
dayStart = zeros(1,size(DATE,1));    

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
end


%Change this time so that 23-07-2007 00:00 be the 0 hour. The time
%is expressed in hours.
clear sunriseL sunriseR solarnoon sunset

SUNRISEL = zeros(1,size(DATE,1));
SUNRISER = zeros(1,size(DATE,1));
solarnoon = zeros(1,size(DATE,1));
sunset = zeros(1,size(DATE,1));
start = zeros(1,size(DATE,1));

for iday = 1:size(DATE,1)
    SUNRISEL(iday) = (iday-1)*24 + daySunriseL(iday);
    SUNRISER(iday) = (iday-1)*24 + daySunriseR(iday);
    solarnoon(iday) = (iday-1)*24 + daySolarnoon(iday);
    sunset(iday) = (iday-1)*24 + daySunset(iday);
    start(iday) = (iday-1)*24 + dayStart(iday);
end

%%%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         SAMPLE = DATA(:,1); %In a file it starts from 0 to 2207 %3534212172:15:3534233757; %Time starts from 00:12 for the first day (23-07-2007) and end at 23:57 on the last day (14-08-2007)() %a(:,1)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         TIME = (0+12:15:2207*15+12)/60; %Time taken from 23-07-2007 00:00. This time is 0. The time is expressed in hours.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %         BRIGHTNESS = DATA(:,2).';

iSunrise = zeros(1,size(DATE,1));
iSunriseL = zeros(1,size(DATE,1));
iSunriseR = zeros(1,size(DATE,1));
iStart = zeros(1,size(DATE,1));
iSolarnoon = zeros(1,size(DATE,1));


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
    SUNRISE_DATA = [SUNRISE_DATA(1); (SUNRISE_DATA(1:end-2) + SUNRISE_DATA(3:end))/2 ; SUNRISE_DATA(end)]; %Moving average
    %sm = 0.25; %scale
    %SUNRISE_DATA = [SUNRISE_DATA(1); (sm/2*SUNRISE_DATA(1:end-2) + (1 - sm)*SUNRISE_DATA(2:end-1) + sm/2*SUNRISE_DATA(3:end));  SUNRISE_DATA(end)]; %FIR smoother
    [dSunriseData iSunriseData]=min(SUNRISE_DATA);
    %TEST_CYCLE_TIME = TIME(CYCLE_LIMIT(1,TEST_FIRE_CYCLE_NUMBER): CYCLE_LIMIT(2,TEST_FIRE_CYCLE_NUMBER))- (TEST_FIRE_CYCLE_NUMBER-1)*24;

    
    iSunrise(iday) = iSunriseL(iday) + iSunriseData(end) - 1;
    
    
    
    iBeforeSunset = find(TIME<=sunset(iday));
    sBeforeSunset = [iBeforeSunset.' TIME(iBeforeSunset).'];
    tBeforeSunset  = sBeforeSunset(:,2);
    [dfBeforeSunset ifBeforeSunset] = max(tBeforeSunset);
    iSunset(iday) = sBeforeSunset(ifBeforeSunset,1);
    
    
    iBeforeSolarnoon = find(TIME<=solarnoon(iday));
    sBeforeSolarnoon = [iBeforeSolarnoon.' TIME(iBeforeSolarnoon).'];
    tBeforeSolarnoon  = sBeforeSolarnoon(:,2);
    [dfBeforeSolarnoon ifBeforeSolarnoon] = max(tBeforeSolarnoon);
    iSolarnoon(iday) = sBeforeSolarnoon(ifBeforeSolarnoon,1);
    
    
    if iday>1
    iBeforeStart = find(TIME<=start(iday));
    sBeforeStart = [iBeforeStart.' TIME(iBeforeStart).'];
    tBeforeStart  = sBeforeStart(:,2);
    [dfBeforeStart ifBeforeStart] = max(tBeforeStart);
    iStart(iday) = sBeforeStart(ifBeforeStart,1);
    else
     iStart(iday) = 1;
    end
end

%Predict sunrise
%[iSunrise morningHorizonTime]=   predictedSunrise_v3(TIME, DATE, iSunrise, longitude, latitude);
[iSunrise morningHorizonTime vtSunrise]=   predictedSunrise_v4(TIME, DATE, iSunrise, longitude, latitude);
vtSunriseDay = vtSunrise(TEST_FIRE_CYCLE_NUMBER);

%Check for thermal rise and correct for these thermal rise
%     for iday = 1:size(DATE,1)
%         rangeThRiseBRIGHTNESS = BRIGHTNESS(iSunrise(iday)-10:iSunrise(iday)+10);
%         rangeThRiseTIME = [iSunrise(iday)-10:iSunrise(iday)+10];
%         dBRIGHTNESS(1,:) = abs(rangeThRiseBRIGHTNESS - [rangeThRiseBRIGHTNESS(2:end); rangeThRiseBRIGHTNESS(length(rangeThRiseBRIGHTNESS))]);
%         dBRIGHTNESS(2,:) = abs(rangeThRiseBRIGHTNESS - [rangeThRiseBRIGHTNESS(1); rangeThRiseBRIGHTNESS(1:end-1)]);
%         [bMinBRIGHTNESS iMinBRIGHTNESS] = max(max(dBRIGHTNESS,[],1));
%         [bMinBRIGHTNESS iMinBRIGHTNESS] = min(rangeThRiseBRIGHTNESS);
%         iSunrise(iday) = rangeThRiseTIME(iMinBRIGHTNESS(length(iMinBRIGHTNESS)));
%     end


%kcycle = [];
%kctime = [];
%cycle = cell(size(DATE,1)-1,100);


%addpath('models');
ctimeK = [];
MODK = [];



w = dayDaylength(TEST_FIRE_CYCLE_NUMBER);

%THE FOLLOWING CODE IS USED TO TEST THE TRAINING
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for iday = 1:size(DATE,1)-1
%     cycle = BRIGHTNESS(iSunrise(iday):iSunrise(iday+1)-1); %Up to 1 sample before SUNRISE
%     ctime = TIME(iSunrise(iday):iSunrise(iday+1)-1);
% 
%     % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             TRAINING_CYCLE = cycle.';
%     TRAINING_CYCLE = cycle;
%     %         if length(cycle)<96
%     %             TRAINING_CYCLE(length(cycle):96) = cycle(length(cycle));
%     %         end
%     TRAINING_CYCLE_TIME = ctime - (iday-1)*24;
%     %TRAINING_CYCLE_TIME = mod(TRAINING_CYCLE_TIME,24);
%     %         if length(ctime)<96
%     %             TRAINING_CYCLE_TIME(length(cycle):96)
% 
%     INIT_ts = daySunset(iday)+2;%+1.7;%+1.7+randn*0.3784;%+2; %+2 hours taken as heating delay
%     INIT_tm = daySolarnoon(iday)+3;%+2.3;%+2.3+randn*0.2480;%+3; %+2 hours taken as heating delay
%     LENGTHOFDAY = dayDaylength(iday);
%     [MOD To Ta w tm ts k dT]=   diurnal_g(TRAINING_CYCLE,LENGTHOFDAY,latitude,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME);    %Gottsche_2001
%     %         [MOD a b td ts beta alpha]= diurnal_j(TRAINING_CYCLE,LENGTHOFDAY,latitude,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME);    %Jiang_2006
%     %         [MOD To Ta tm ts k dT tau]= diurnal_g_z(TRAINING_CYCLE,INIT_tm,INIT_ts,TRAINING_CYCLE_TIME,year,month,day,longitude,latitude); %Gottsche_2009
%     %         [MOD To Ta tm ts k dT tau]= diurnal_g3(TRAINING_CYCLE,INIT_tm-3,INIT_ts-2,TRAINING_CYCLE_TIME,year,month,day,longitude,latitude); %Gottsche_2009
%     %Train with more cycles and use chi-square fitting and other
%     %approaches such as MSE and so on.
%     %Call also robust fitting and RKHS models and Göttsche_2009 and Frans model and Shadlick_1999
%     %         [To Ta tm ts k dT tau]
% 
%     %         format long
%     %         tau
%     %         format
%     %kcycle = [kcycle cycle];
%     %kctime = [kctime ctime];
% 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %length(cycle)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     plot(ctime,cycle,'r*')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     hold on
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     plot(ctime,real(MOD))
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     hold off
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     legend('Valid observed samples','Interpolated curve')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     xlabel('Time stamp')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     ylabel('Brightness temperature (Kelvin)')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     title('Fitting from the training')
%     %title('TRAINING')
%     %DTR(iday) = max(cycle) - min(cycle);
%     MODK = [MODK MOD];
%     ctimeK = [ctimeK ctime];
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % figure
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % plot(ctimeK,MODK)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % grid
%plot(kctime,kcycle)
%grid


CYCLE_LIMIT = [];
for iday = 1:size(DATE,1)-1
    LIMIT = [iSunrise(iday)+1;iSunrise(iday+1)];
    CYCLE_LIMIT = [CYCLE_LIMIT LIMIT];
end


%iday
%TEST_FIRE_CYCLE_NUMBER

TEST_FIRE_CYCLE = BRIGHTNESS(CYCLE_LIMIT(1,TEST_FIRE_CYCLE_NUMBER): CYCLE_LIMIT(2,TEST_FIRE_CYCLE_NUMBER)).';
TEST_CYCLE_TIME = TIME(CYCLE_LIMIT(1,TEST_FIRE_CYCLE_NUMBER): CYCLE_LIMIT(2,TEST_FIRE_CYCLE_NUMBER))- (TEST_FIRE_CYCLE_NUMBER-1)*24;
%TEST_CYCLE_TIME = mod(TEST_CYCLE_TIME,24);


LENGTH_CURRENT_CYCLE = length(TEST_FIRE_CYCLE);
%CURRENT_CYLE = TEST_FIRE_CYCLE_NUMBER;
%[To_M Ta_M tm_M ts_M dT_M Q R1 bA MNs mp cycle_out_scope] = noisemodel(current_txt_file,TEST_FIRE_CYCLE_NUMBER,latitude, longitude,LENGTH_CURRENT_CYCLE);
%[MOD,To_M Ta_M tm_M ts_M dT_M Q R1 bA MNs mp cycle_out_scope] = DTCgraph2(current_txt_file,TEST_FIRE_CYCLE_NUMBER,latitude, longitude,LENGTH_CURRENT_CYCLE);
[MOD,To_M Ta_M tm_M ts_M dT_M w_M k_M Q R1 bA MNs mp cycle_out_scope w2_M tau_M m_z_M P_M v_z_M v_m_M theta_z_M theta_zm_M theta_zs_M m_zs_M tsr_P_M a_P_M Y_P_M Z_P_M a_RKHS_M K_RKHS_M deltaK_RKHS_M a_SVD_M U_SVD_M deltaU_SVD_M ITERATIONS_A FCTCOUNT_A RMSE_NSQ_OVER_MISSING_VALUE_ONLY MAE_NSQ_OVER_MISSING_VALUE_ONLY BIAS_NSQ_OVER_MISSING_VALUE_ONLY tmin_M horizon missingdataFlag] = DTCModel_Noise(current_txt_file,TEST_FIRE_CYCLE_NUMBER,latitude, longitude,LENGTH_CURRENT_CYCLE,MODEL,ErrFct,ACTIVATE_MISSINGSAMPLE,GAP,rw,cm,DDTP_Nbre_Cycles);


if length(w_M) ~= 0
    w = w_M;
end


    
%LENGTH_CURRENT_CYCLE
%length(MOD)

%plot(TEST_CYCLE_TIME, MOD,'r')
%             %REPORT UNDECIDED
%w = w_M
%TEST_FIRE_CYCLE_NUMBER



tSunriseDay = TIME(iSunrise(TEST_FIRE_CYCLE_NUMBER)) - (TEST_FIRE_CYCLE_NUMBER-1)*24;
tSunsetDay = TIME(iSunset(TEST_FIRE_CYCLE_NUMBER)) - (TEST_FIRE_CYCLE_NUMBER-1)*24;
tStartDay = TIME(iStart(TEST_FIRE_CYCLE_NUMBER)) - (TEST_FIRE_CYCLE_NUMBER-1)*24;
tDayLength = dayDaylength(TEST_FIRE_CYCLE_NUMBER);
%tSolarNoon = solarnoon(TEST_FIRE_CYCLE_NUMBER) - (TEST_FIRE_CYCLE_NUMBER-1)*24;
tSolarnoonDay = TIME(iSolarnoon(TEST_FIRE_CYCLE_NUMBER)) - (TEST_FIRE_CYCLE_NUMBER-1)*24;
tmorningHorizon = morningHorizonTime(TEST_FIRE_CYCLE_NUMBER) - (TEST_FIRE_CYCLE_NUMBER-1)*24;

SunriseDay = iSunrise(TEST_FIRE_CYCLE_NUMBER) - iSunrise(8) + 1;
SunsetDay = iSunset(TEST_FIRE_CYCLE_NUMBER) - iSunrise(8) + 1;
StartDay = iStart(TEST_FIRE_CYCLE_NUMBER) - iSunrise(8) + 1;


sr = [tSunriseDay tsr_P_M]; %Thermal sunrise calculated and the one given by Parton1981

%iStart
%iSunrise
%iSunset
%Test DDTP:
if MODEL == 'u' %DDTP (Duan_2013)
    TEST_FIRE_CYCLE = BRIGHTNESS(CYCLE_LIMIT(1,TEST_FIRE_CYCLE_NUMBER): CYCLE_LIMIT(2,TEST_FIRE_CYCLE_NUMBER + DDTP_Nbre_Cycles - 1)).';
    TEST_CYCLE_TIME = [];
    for iDT = 0:DDTP_Nbre_Cycles-1
        TEST_CYCLE_TIME = [TEST_CYCLE_TIME TIME(CYCLE_LIMIT(1,TEST_FIRE_CYCLE_NUMBER + iDT): CYCLE_LIMIT(2,TEST_FIRE_CYCLE_NUMBER + iDT))- (TEST_FIRE_CYCLE_NUMBER + iDT -1)*24];
    end
end
