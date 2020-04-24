function [TESTINGCYCLE_DATA, CHECK_LENGTH, NUMBER_ROW, NUMBER_COLUMN, INDEX_ROW, INDEX_COLUMN KEEP_TIME] = modelParameters(CENTRE_ROW,CENTRE_COLUMN,RIGHT_CYCLE, FNDATAREGION, S_COLUMN, E_COLUMN, S_ROW, E_ROW, S_CYCLE)
    
    N_LINES = (E_ROW - S_ROW) + 1;
    N_COLUMNS = (E_COLUMN - S_COLUMN) + 1;
    KeepINDEXROWCOLUM = [];
    
    %S_CYCLE = 8;
    %RIGHT_CYCLE = 8:10;
    
    
    % close all
    % clear all
    % clc
    
    %In this file load FNDATAREGIONFILE is not done, comments also the above.
    %CENTRE_ROW = 2749; %1014; %1076; %1076; %1018; %1030; %1050; %1035; %860; %1060; %930; %917; %1018;   %851; %1018; %851;  %894; %898; %904;  %898  894
    %CENTRE_COLUMN = 942; %984; %941; %947;%886; %990; %884; %878; %1000; %930; %963; %979; %886; %966;  %886; %966;  %976; %980; %1074; %980  976
    %RIGHT_CYCLE = 8:13;%11; %8; %11; %10;
    
    
    MODEL = 'v';
    ERRFUNCTION = 'R'; %L:Least squares, C:Chi-square error, R:Robust error
    NONSEQUENTIAL = 1; %Not sequential = 1, sequential = 0
    DDTP_Nbre_Cycles = 1;
    
    % Excel_Potential.Quit %Quit the application
    % Excel_Potential.delete %Delete the object
    % clear Excel_Potential
    
    
    ACTIVATE_MISSINGSAMPLE = 0; %1: Simulate missing data, 0:No simulation of missing data
    GAP = 1; %GAP: one of the three gap of missing data simulated.
    
    if ACTIVATE_MISSINGSAMPLE == 1
        display('Missing data simulation');
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%
    
    
    % FILE_LOCATION = 'C:\Users\juwilingiye\UJ_PREDICTION1\';
    % EXTENDED_POTENTIAL_FILE = strcat(FILE_LOCATION,'MODEL_RESULTS\xtpotential');
    % % FILENAMEPART = EXTENDED_POTENTIAL_FILE;
    %
    % Excel_Potential = actxserver ('Excel.Application'); %Open connection to Excel
    %
    %
    % FILE_NUMBER_POTENTIAL = 1; %Initialize the first xls file to writen to, it will have a suffix of 1.
    % XLS_ROW_NUMBER_POTENTIAL = 1;
    START_ROW = 778;
    START_COLUMN = 822;
    % EXCEL = 'Excel';
    
    
    %if NONSEQUENTIAL ==1
    % addpath('Eigentracking');
    % addpath('LMFnlsq');
    % addpath('LMFnlsq2_220413');
    % addpath('LMFsolve');
    % addpath('Powell');
    % addpath('C:\Users\JOSINE\WORK1\work\FireDetection2\models');
    % addpath('C:\Users\JOSINE\WORK1\work\FireDetection2\solarTimeLocation');
    % addpath('C:\Users\JOSINE\WORK1\work\FireDetection2\TimePrediction');
    % addpath('C:\Users\JOSINE\WORK1\work\NDeconvolution')
    % addpath('..\DTCModel')
    
    %load FNDATAREGIONFILE;
    disp('started')
    
    % WINDOWSIZE = 1;
    % W_CENTRE = ceil(WINDOWSIZE/2);
    
    NBRE_LINES = N_LINES;
    NBRE_COLUMNS = N_COLUMNS;
    
    TESTINGCYCLE_DATA(NBRE_LINES,NBRE_COLUMNS).TEST_CYCLE_DATA = [];
    TESTINGCYCLE_DATA(1,1).TEST_CYCLE_TIME = [];
    TESTINGCYCLE_DATA(1,1).w = [];
    TESTINGCYCLE_DATA(1,1).To = [];
    TESTINGCYCLE_DATA(1,1).Ta = [];
    TESTINGCYCLE_DATA(1,1).tm = [];
    TESTINGCYCLE_DATA(1,1).ts = [];
    TESTINGCYCLE_DATA(1,1).dT = [];
    TESTINGCYCLE_DATA(1,1).k = [];
    TESTINGCYCLE_DATA(1,1).Q = [];
    TESTINGCYCLE_DATA(1,1).R1 = [];
    TESTINGCYCLE_DATA(1,1).R1c = [];
    TESTINGCYCLE_DATA(1,1).TREATED = [];
    TESTINGCYCLE_DATA(1,1).w2 = [];
    
    %TESTINGCYCLE_DATA(1,1).tsr_P_M = [];
    TESTINGCYCLE_DATA(1,1).SunriseDay = [];
    %TESTINGCYCLE_DATA(1,1).StartDay = [];
    TESTINGCYCLE_DATA(1,1).tSunriseDay = [];
    %TESTINGCYCLE_DATA(1,1).tStartDay = [];
    %TESTINGCYCLE_DATA(1,1).sr = [];
    %TESTINGCYCLE_DATA(1,1).tmin_M = [];
    TESTINGCYCLE_DATA(1,1).tmorningHorizon = [];
    TESTINGCYCLE_DATA(1,1).tSolarNoon = [];
    TESTINGCYCLE_DATA(1,1).tSunsetDay = [];
    
    TESTINGCYCLE_DATA(1,1).LEFT_CYCLE_SAMPLE = [];
    TESTINGCYCLE_DATA(1,1).CovParameter = [];
    TESTINGCYCLE_DATA(1,1).CovParameterN = [];
    
    % CENTRE_COLUMN = 886; %966;  %886; %966;  %976; %980; %1074; %980  976
    % CENTRE_ROW = 1018;   %851; %1018; %851;  %894; %898; %904;  %898  894
    
    % COLUMN:    966 1074 980 976
    % ROW:       851  904 898 894
    % TIMESTAMP: 221  211 222 222
    
    KEEP_MOD = [];
    KEEP_START = [];
    KEEP_SUNRISE = [];
    KEEP_SUNSET = [];
    KEEP_tDAYLENGTH = [];
    KEEP_tSOLARNOON = [];
    KEEP_tSTART = [];
    KEEP_missingdataFlag = [];
    KEEP_TIME = [];
    
    
    START_CYCLE = S_CYCLE; %USE TIME INSTEAD OF CYCLE, AND TRAIN FOR A SINGLE PIXEL (noisemodel), CONSIDER THAT PIXELS HAVE DIFFERENT SAMPLE SIZE FOR A GIVEN CYCLE, TRAINING MUST BE IMPLEMENTED AT DIFFERENT TIME TO GET PARAMETERS, AND REDUCE THE SIZE OF THE STRUCTURES THAT KEEP PARAMETERS AT EACH TIME A CYCLE OF ONE OF THE PIXEL ENDED
    for TEST_FIRE_CYCLE_NUMBER = RIGHT_CYCLE  %9:9%8:13 %8:14 %8 is the cycle starting from 30-07 sunrise and end at 31-07 before sunrise
        START_COLUMN = S_COLUMN %CENTRE_COLUMN; %980-2 %1074-2 %966-2 %157+822-1;
        END_COLUMN = E_COLUMN %CENTRE_COLUMN;
        for COLUMN = START_COLUMN:END_COLUMN %980+2 %1074+2 %966+2 %161+822-1  %44+822-1:54+822-1    %1000:1000     %881:881    %886:886  %929:929     %929:929  %44+822-1:54+822-1  %929:929    %1000:1000%54+822-1:54+822-1    %44+822-1:44+822-1 %886:886 -> KRUGER PARK (NOFIRE)   %61+822-1:61+822-1 -> KRUGER PARK (FIRE) %1000:1000 -> NOFIRE %942:942 ->FIRE %1000:1000%942:942  %822:1094 COLUMN
            START_ROW = S_ROW %CENTRE_ROW; %898-2 %904-2 %851-2 %119+778-1;
            END_ROW = E_ROW %CENTRE_ROW;
            for ROW = START_ROW:END_ROW %898+2 %904+2 %851+2 %123+778-1 %283+778-1:293+778-1 %1000:1000  %1064:1064 %972:972 %1065:1065  %963:963 %1065:1065%283+778-1:293+778-1 %1065:1065 %1000:1000%283+778-1:283+778-1   %283+778-1:283+778-1 %972:972 %199+778-1:199+778-1%1000:1000%963:963 %1000:1000%963:963 %778:1153 LINE
                
                %FIRE MASK BEFORE MODELLING SO THAT A SAMPLE CORRUPTED BY FIRE
                %IS NOT USED IN TRAINING
                %figure
                %hold on
                
                %[MOD,TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude,longitude,column, row,To_M, Ta_M, tm_M, ts_M, dT_M, Q, R1, bA, MNs, mp, cycle_out_scope]= CycleLocationTraining(COLUMN,ROW,TEST_FIRE_CYCLE_NUMBER,FNDATAREGION);
                %[MOD,TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude,longitude,column, row,To_M, Ta_M, tm_M, ts_M, dT_M, Q, R1, bA, MNs, mp, cycle_out_scope]= CycleLocationTraining2(COLUMN,ROW,TEST_FIRE_CYCLE_NUMBER,FNDATAREGION);
                %[MOD,TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude,longitude,column, row,To_M, Ta_M, tm_M, ts_M, dT_M, k_M, w2_M, Q, R1, bA, MNs, mp, cycle_out_scope,SunriseDay, SunsetDay, StartDay, tSunriseDay, tSunsetDay, tStartDay, tDayLength, tSolarNoon]= CycleLocationTraining3_(COLUMN,ROW,TEST_FIRE_CYCLE_NUMBER,FNDATAREGION);
                %[MOD,TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude,longitude,column, row,To_M, Ta_M, tm_M, ts_M, dT_M, k_M, w2_M, tau_M, m_z_M, P_M, v_z_M, v_m_M, theta_z_M, theta_zm_M, theta_zs_M, m_zs_M, tsr_P_M, a_P_M, Y_P_M, Z_P_M, a_RKHS_M, K_RKHS_M, deltaK_RKHS_M, a_SVD_M U_SVD_M deltaU_SVD_M Q, R1, bA, MNs, mp, cycle_out_scope,SunriseDay, SunsetDay, StartDay, tSunriseDay, tSunsetDay, tStartDay, tDayLength, tSolarNoon sr ITERATIONS_A FCTCOUNT_A]= CycleLocationTraining3_(COLUMN,ROW,TEST_FIRE_CYCLE_NUMBER,FNDATAREGION,MODEL,ERRFUNCTION);
                %[MOD,TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude,longitude,column, row,To_M, Ta_M, tm_M, ts_M, dT_M, k_M, w2_M, tau_M, m_z_M, P_M, v_z_M, v_m_M, theta_z_M, theta_zm_M, theta_zs_M, m_zs_M, tsr_P_M, a_P_M, Y_P_M, Z_P_M, a_RKHS_M, K_RKHS_M, deltaK_RKHS_M, a_SVD_M U_SVD_M deltaU_SVD_M Q, R1, bA, MNs, mp, cycle_out_scope,SunriseDay, SunsetDay, StartDay, tSunriseDay, tSunsetDay, tStartDay, tDayLength, tSolarNoon sr ITERATIONS_A FCTCOUNT_A RMSE_NSQ_OVER_MISSING_VALUE_ONLY MAE_NSQ_OVER_MISSING_VALUE_ONLY BIAS_NSQ_OVER_MISSING_VALUE_ONLY]= CycleLocationTraining3_(COLUMN,ROW,TEST_FIRE_CYCLE_NUMBER,FNDATAREGION,MODEL,ERRFUNCTION,ACTIVATE_MISSINGSAMPLE,GAP);
                %[MOD,TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude,longitude,column, row,To_M, Ta_M, tm_M, ts_M, dT_M, k_M, w2_M, tau_M, m_z_M, P_M, v_z_M, v_m_M, theta_z_M, theta_zm_M, theta_zs_M, m_zs_M, tsr_P_M, a_P_M, Y_P_M, Z_P_M, a_RKHS_M, K_RKHS_M, deltaK_RKHS_M, a_SVD_M U_SVD_M deltaU_SVD_M Q, R1, bA, MNs, mp, cycle_out_scope,SunriseDay, SunsetDay, StartDay, tSunriseDay, tSunsetDay, tStartDay, tDayLength, tSolarNoon sr ITERATIONS_A FCTCOUNT_A RMSE_NSQ_OVER_MISSING_VALUE_ONLY MAE_NSQ_OVER_MISSING_VALUE_ONLY BIAS_NSQ_OVER_MISSING_VALUE_ONLY tmin_M horizon]= CycleLocationTraining3_(COLUMN,ROW,TEST_FIRE_CYCLE_NUMBER,FNDATAREGION,MODEL,ERRFUNCTION,ACTIVATE_MISSINGSAMPLE,GAP, DDTP_Nbre_Cycles);
                
                %[MOD,TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude,longitude,column, row,To_M, Ta_M, tm_M, ts_M, dT_M, k_M, w2_M, tau_M, m_z_M, P_M, v_z_M, v_m_M, theta_z_M, theta_zm_M, theta_zs_M, m_zs_M, tsr_P_M, a_P_M, Y_P_M, Z_P_M, a_RKHS_M, K_RKHS_M, deltaK_RKHS_M, a_SVD_M U_SVD_M deltaU_SVD_M Q, R1, bA, MNs, mp, cycle_out_scope,SunriseDay, SunsetDay, StartDay, tSunriseDay, tSunsetDay, tStartDay, tDayLength, tSolarNoon sr ITERATIONS_A FCTCOUNT_A RMSE_NSQ_OVER_MISSING_VALUE_ONLY MAE_NSQ_OVER_MISSING_VALUE_ONLY BIAS_NSQ_OVER_MISSING_VALUE_ONLY tmin_M horizon missingdataFlag]= CycleLocationTraining9(COLUMN,ROW,TEST_FIRE_CYCLE_NUMBER,FNDATAREGION,MODEL,ERRFUNCTION,ACTIVATE_MISSINGSAMPLE,GAP, DDTP_Nbre_Cycles);
                
                
                %OWN PARAMETERS
                %[MOD,TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude,longitude,column, row,To_M, Ta_M, tm_M, ts_M, dT_M, k_M, w2_M, tau_M, m_z_M, P_M, v_z_M, v_m_M, theta_z_M, theta_zm_M, theta_zs_M, m_zs_M, tsr_P_M, a_P_M, Y_P_M, Z_P_M, a_RKHS_M, K_RKHS_M, deltaK_RKHS_M, a_SVD_M U_SVD_M deltaU_SVD_M Q, R1, bA, MNs, mp, cycle_out_scope,SunriseDay, SunsetDay, StartDay, tSunriseDay, tSunsetDay, tStartDay, tDayLength, tSolarNoon sr ITERATIONS_A FCTCOUNT_A RMSE_NSQ_OVER_MISSING_VALUE_ONLY MAE_NSQ_OVER_MISSING_VALUE_ONLY BIAS_NSQ_OVER_MISSING_VALUE_ONLY tmin_M horizon missingdataFlag]= CycleLocationTraining9(COLUMN,ROW,TEST_FIRE_CYCLE_NUMBER,FNDATAREGION,MODEL,ERRFUNCTION,ACTIVATE_MISSINGSAMPLE,GAP, DDTP_Nbre_Cycles);
                
                %MEAN OF PARAMETERS
                %[MOD,TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude,longitude,column, row,To_M, Ta_M, tm_M, ts_M, dT_M, k_M, w2_M, tau_M, m_z_M, P_M, v_z_M, v_m_M, theta_z_M, theta_zm_M, theta_zs_M, m_zs_M, tsr_P_M, a_P_M, Y_P_M, Z_P_M, a_RKHS_M, K_RKHS_M, deltaK_RKHS_M, a_SVD_M U_SVD_M deltaU_SVD_M Q, R1, bA, MNs, mp, cycle_out_scope,SunriseDay, SunsetDay, StartDay, tSunriseDay, tSunsetDay, tStartDay, tDayLength, tSolarNoon sr ITERATIONS_A FCTCOUNT_A RMSE_NSQ_OVER_MISSING_VALUE_ONLY MAE_NSQ_OVER_MISSING_VALUE_ONLY BIAS_NSQ_OVER_MISSING_VALUE_ONLY tmin_M horizon missingdataFlag tmorningHorizon CovParameter NTRAININGCYCLE]= CycleLocationTraining10_v4(COLUMN,ROW,TEST_FIRE_CYCLE_NUMBER,FNDATAREGION,MODEL,ERRFUNCTION,ACTIVATE_MISSINGSAMPLE,GAP, DDTP_Nbre_Cycles);
                %[MOD,TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude,longitude,column, row,To_M, Ta_M, tm_M, ts_M, dT_M, k_M, w2_M, tau_M, m_z_M, P_M, v_z_M, v_m_M, theta_z_M, theta_zm_M, theta_zs_M, m_zs_M, tsr_P_M, a_P_M, Y_P_M, Z_P_M, a_RKHS_M, K_RKHS_M, deltaK_RKHS_M, a_SVD_M U_SVD_M deltaU_SVD_M Q, R1, bA, MNs, mp, cycle_out_scope,SunriseDay, SunsetDay, StartDay, tSunriseDay, tSunsetDay, tStartDay, tDayLength, tSolarNoon sr ITERATIONS_A FCTCOUNT_A RMSE_NSQ_OVER_MISSING_VALUE_ONLY MAE_NSQ_OVER_MISSING_VALUE_ONLY BIAS_NSQ_OVER_MISSING_VALUE_ONLY tmin_M horizon missingdataFlag tmorningHorizon]= CycleLocationTraining10_v3(COLUMN,ROW,TEST_FIRE_CYCLE_NUMBER,FNDATAREGION,MODEL,ERRFUNCTION,ACTIVATE_MISSINGSAMPLE,GAP, DDTP_Nbre_Cycles);
                [MOD,TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude,longitude,column, row,To_M, Ta_M, tm_M, ts_M, dT_M, k_M, w2_M, tau_M, m_z_M, P_M, v_z_M, v_m_M, theta_z_M, theta_zm_M, theta_zs_M, m_zs_M, tsr_P_M, a_P_M, Y_P_M, Z_P_M, a_RKHS_M, K_RKHS_M, deltaK_RKHS_M, a_SVD_M U_SVD_M deltaU_SVD_M Q, R1, bA, MNs, mp, cycle_out_scope,SunriseDay, SunsetDay, StartDay, tSunriseDay, tSunsetDay, tStartDay, tDayLength, tSolarNoon sr ITERATIONS_A FCTCOUNT_A RMSE_NSQ_OVER_MISSING_VALUE_ONLY MAE_NSQ_OVER_MISSING_VALUE_ONLY BIAS_NSQ_OVER_MISSING_VALUE_ONLY tmin_M horizon missingdataFlag tmorningHorizon vtSunriseDay]= CycleLocationTraining10_v5(COLUMN,ROW,TEST_FIRE_CYCLE_NUMBER,FNDATAREGION,MODEL,ERRFUNCTION,ACTIVATE_MISSINGSAMPLE,GAP, DDTP_Nbre_Cycles);
                
                
                KEEP_MOD = [KEEP_MOD MOD];
                KEEP_missingdataFlag = [KEEP_missingdataFlag missingdataFlag.'];
                KEEP_TIME = [KEEP_TIME, TEST_CYCLE_TIME + (TEST_FIRE_CYCLE_NUMBER-8)*24;];
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             KEEP_START = [KEEP_START StartDay];
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             KEEP_SUNRISE = [KEEP_SUNRISE SunriseDay];
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             KEEP_SUNSET = [KEEP_SUNSET SunsetDay];
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             KEEP_tDAYLENGTH = [KEEP_tDAYLENGTH tDayLength];
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             KEEP_tSOLARNOON = [KEEP_tSOLARNOON tSolarNoon];
                % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %             KEEP_tSTART = [KEEP_tSTART tStartDay];
                %, tSunriseDay, tSunsetDay, tStartDay
                
                %plot(TEST_CYCLE_TIME, MOD,'r')
                
                %hold off
                %bA and MNs may contains some NaN even with few cycles to use for
                %training (cycle<7), other variables will be NaN if there no
                %cycle (even one) to use for training. cycle_out_scope can be 1
                %or 0, not NaN. It is 1 to show that the cycle used were less
                %than 7 or 0 to show that the used number of cycles are 7.
                
                INDEX_ROW = ROW - START_ROW + 1;
                INDEX_COLUMN = COLUMN - START_COLUMN + 1;
                INDEX_CYCLE = TEST_FIRE_CYCLE_NUMBER - START_CYCLE + 1;
                
                
                
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA TEST_FIRE_CYCLE];
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_TIME = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_TIME TEST_CYCLE_TIME];
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_NUMBER(INDEX_CYCLE) = TEST_FIRE_CYCLE_NUMBER; %MUST BE REMOVED
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_LENGTH(INDEX_CYCLE) = length(TEST_FIRE_CYCLE); %MUST BE REMOVED
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).LATITUDE(INDEX_CYCLE) = latitude;
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).LONGITUDE(INDEX_CYCLE) = longitude;
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).ROW(INDEX_CYCLE) = row;  %CAN BE REMOVED
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).COLUMN(INDEX_CYCLE) = column; %CAN BE REMOVED
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).w = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).w ones(1,length(TEST_FIRE_CYCLE))*w];
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).w2 = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).w2 ones(1,length(TEST_FIRE_CYCLE))*w2_M];
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).To = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).To ones(1,length(TEST_FIRE_CYCLE))*To_M];
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).Ta = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).Ta ones(1,length(TEST_FIRE_CYCLE))*Ta_M];
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).tm = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).tm ones(1,length(TEST_FIRE_CYCLE))*tm_M];
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).ts = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).ts ones(1,length(TEST_FIRE_CYCLE))*ts_M];
                %TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).dT = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).dT ones(1,length(TEST_FIRE_CYCLE))*dT_M];
                %k_M = w2_M/pi.*(1./tan(pi./w2_M.*(ts_M - tm_M)));
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).k = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).k ones(1,length(TEST_FIRE_CYCLE))*k_M];
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).Q = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).Q Q.'];
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).R1 = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).R1 ones(1,length(TEST_FIRE_CYCLE))*R1];
                
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).R1c = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).R1c ones(1,length(TEST_FIRE_CYCLE))*vtSunriseDay];
                
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TREATED = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TREATED ones(1,length(TEST_FIRE_CYCLE))*cycle_out_scope];
                
                %SSY = SunriseDay
                %TSY = tSunriseDay
                
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).SunriseDay = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).SunriseDay ones(1,length(TEST_FIRE_CYCLE))*SunriseDay];
                %TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).StartDay = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).StartDay ones(1,length(TEST_FIRE_CYCLE))*StartDay];
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).tSunriseDay = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).tSunriseDay ones(1,length(TEST_FIRE_CYCLE))*tSunriseDay];
                %TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).tStartDay = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).tStartDay ones(1,length(TEST_FIRE_CYCLE))*tStartDay];
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).tmorningHorizon = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).tmorningHorizon ones(1,length(TEST_FIRE_CYCLE))*tmorningHorizon];
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).tSolarNoon = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).tSolarNoon ones(1,length(TEST_FIRE_CYCLE))*tSolarNoon];
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).tSunsetDay = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).tSunsetDay ones(1,length(TEST_FIRE_CYCLE))*tSunsetDay];
                
                TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).LEFT_CYCLE_SAMPLE = [TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).LEFT_CYCLE_SAMPLE (length(TEST_FIRE_CYCLE):-1:1)-1]; %MUST BE REMOVED
                
                if  TEST_FIRE_CYCLE_NUMBER == RIGHT_CYCLE(1)
                    %[MOD,TEST_FIRE_CYCLE,w,TEST_CYCLE_TIME,TEST_FIRE_CYCLE_NUMBER,latitude,longitude,col1, rw1,To_M, Ta_M, tm_M, ts_M, dT_M, k_M, w2_M, tau_M, m_z_M, P_M, v_z_M, v_m_M, theta_z_M, theta_zm_M, theta_zs_M, m_zs_M, tsr_P_M, a_P_M, Y_P_M, Z_P_M, a_RKHS_M, K_RKHS_M, deltaK_RKHS_M, a_SVD_M U_SVD_M deltaU_SVD_M Q, R1, bA, MNs, mp, cycle_out_scope,SunriseDay, SunsetDay, StartDay, tSunriseDay, tSunsetDay, tStartDay, tDayLength, tSolarNoon sr ITERATIONS_A FCTCOUNT_A RMSE_NSQ_OVER_MISSING_VALUE_ONLY MAE_NSQ_OVER_MISSING_VALUE_ONLY BIAS_NSQ_OVER_MISSING_VALUE_ONLY tmin_M horizon missingdataFlag tmorningHorizon CovParameter NTRAININGCYCLE]= CycleLocationTraining10_vCov(COLUMN,ROW,RIGHT_CYCLE(end),FNDATAREGION,MODEL,ERRFUNCTION,ACTIVATE_MISSINGSAMPLE,GAP, DDTP_Nbre_Cycles);
                    %[CovParameter NTRAININGCYCLE]= CycleLocationTraining10_vCov(COLUMN,ROW,13,FNDATAREGION,MODEL,ERRFUNCTION,ACTIVATE_MISSINGSAMPLE,GAP, DDTP_Nbre_Cycles);
                    fParameter = CycleLocationTraining10_vCov2(COLUMN,ROW,13,FNDATAREGION,MODEL,ERRFUNCTION,ACTIVATE_MISSINGSAMPLE,GAP, DDTP_Nbre_Cycles);
                    
                    MParameter = nanmean(fParameter,2);
                    CovParameter = 0;
                    for i = 1:size(fParameter,2)
                        if sum(isnan(fParameter(:,i)))==0
                            CovParameter = CovParameter + (fParameter(:,i)-MParameter)*(fParameter(:,i)-MParameter).';
                        end
                    end
                    NTRAININGCYCLE = sum(~isnan(fParameter(1,:)));
                    CovParameter = CovParameter/(NTRAININGCYCLE-1); %(size(fParameter,2)-1);
                    
                    TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).CovParameter = CovParameter; %diag(diag(CovParameter)); %; %diag(diag(CovParameter)); %10*eye(6)
                    NTC(INDEX_ROW,INDEX_COLUMN) = NTRAININGCYCLE;
                    %IIII_(INDEX_ROW,INDEX_COLUMN) = 1;
                    %KeepINDEXROWCOLUM = [KeepINDEXROWCOLUM; INDEX_ROW INDEX_COLUMN];
                    
                    
                    %In the neighbourhood of the pixel concerned
                    neighbourhood = {[-1,-1], [-1,0], [-1,1], [0,-1], [0,1], [1,-1], [1,0], [1,1]};
                    %neighbourhood = {[-1,-1], [-1,0], [-1,1], [0,-1], [0,1], [1,-1], [1,0], [1,1], [-2,-2], [-2,-1], [-2,0], [-2,1], [-2,2], [-1,-2], [0,-2], [1,-2], [2,-2], [-1,2], [0,2], [1,2], [2,-1], [2,0], [2,1], [2,2]};
                    CovParameterp = (NTRAININGCYCLE - 1)*CovParameter;
                    NTRAININGCYCLETOTAL = NTRAININGCYCLE;
                    for dir = 1:8 %24 %8 %24 %8
                        %                         fParameter = [fParameter CycleLocationTraining10_vCov2(COLUMN+neighbourhood{dir}(2),ROW+neighbourhood{dir}(1),13,FNDATAREGION,MODEL,ERRFUNCTION,ACTIVATE_MISSINGSAMPLE,GAP, DDTP_Nbre_Cycles)];
                        fParameter = CycleLocationTraining10_vCov2(COLUMN+neighbourhood{dir}(2),ROW+neighbourhood{dir}(1),13,FNDATAREGION,MODEL,ERRFUNCTION,ACTIVATE_MISSINGSAMPLE,GAP, DDTP_Nbre_Cycles);
                        MParameter = nanmean(fParameter,2);
                        CovParameter = 0;
                        for i = 1:size(fParameter,2)
                            if sum(isnan(fParameter(:,i)))==0
                                CovParameter = CovParameter + (fParameter(:,i)-MParameter)*(fParameter(:,i)-MParameter).';
                            end
                        end
                        NTRAININGCYCLE = sum(~isnan(fParameter(1,:)));
                        CovParameter = CovParameter/(NTRAININGCYCLE-1); %(size(fParameter,2)-1);
                        
                        CovParameterp = CovParameterp + (NTRAININGCYCLE - 1)*CovParameter;
                        NTRAININGCYCLETOTAL = NTRAININGCYCLETOTAL + NTRAININGCYCLE;
                    end
                    CovParameterp = CovParameterp/(NTRAININGCYCLETOTAL - 9);
                    
                    TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).CovParameter = CovParameterp; %diag(diag(CovParameter)); %CovParameter;
                    NTC(INDEX_ROW,INDEX_COLUMN) = NTRAININGCYCLETOTAL;
                    %                     TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).CovParameterN = CovParameter; %diag(diag(CovParameter)); %CovParameter;
                    %                     NTC(INDEX_ROW,INDEX_COLUMN) = NTRAININGCYCLE;
                    
                    %MECS (Maximum entropy covariance estimate) by Thomaz et al.,2004
                    %                     [VMECS DMECS] = eig(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).CovParameter+TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).CovParameterN);
                    %                     viMECS = diag(VMECS'*TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).CovParameter*VMECS);
                    %                     vpMECS = diag(VMECS'*TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).CovParameterN*VMECS);
                    %                     %viMECS = eig(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).CovParameter);
                    %                     %vpMECS = eig(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).CovParameterN);
                    %                     vzMECS = diag(max(viMECS,vpMECS));
                    %                     TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).CovParameter = VMECS*vzMECS*VMECS';
                    
                end
                
                
                
                %calculate
                %residual
                %RMSE
                %MAD
                %BIAS
                
                
                
                %RESIDUAL = MOD - TEST_FIRE_CYCLE;
                %RMSE_NSQ = sqrt(mean(RESIDUAL.^2));
                %MAE_NSQ = mean(abs(RESIDUAL));
                %BIAS_NSQ = sum(RESIDUAL);
                
                %OBSERVATION = TEST_FIRE_CYCLE;
                %PREDICTION = MOD;
                %TIME = TEST_CYCLE_TIME;
                
                %ts_k = TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).ts;
                
                %save TIME_BER_TM_UP TIME OBSERVATION PREDICTION RESIDUAL RMSE_NSQ MAE_NSQ BIAS_NSQ
                %save TIME_BER_TM_DOWN TIME OBSERVATION PREDICTION RESIDUAL RMSE_NSQ MAE_NSQ BIAS_NSQ
                %save TIME_BER_TTSR_XT TIME OBSERVATION PREDICTION RESIDUAL RMSE_NSQ MAE_NSQ BIAS_NSQ
                %save TIME_BER_TTSR_FS TIME OBSERVATION PREDICTION RESIDUAL RMSE_NSQ MAE_NSQ BIAS_NSQ
                %save TIME_BER_ts TIME OBSERVATION PREDICTION RESIDUAL RMSE_NSQ MAE_NSQ BIAS_NSQ ts_k
                %save TIME_BER_night TIME OBSERVATION PREDICTION RESIDUAL RMSE_NSQ MAE_NSQ BIAS_NSQ ts_k
                %save TIME_BER_Gap2 TIME OBSERVATION PREDICTION RESIDUAL RMSE_NSQ MAE_NSQ BIAS_NSQ RMSE_NSQ_OVER_MISSING_VALUE_ONLY MAE_NSQ_OVER_MISSING_VALUE_ONLY BIAS_NSQ_OVER_MISSING_VALUE_ONLY ts_k KEEP_missingdataFlag
                %save TIME_BER_Gap3 TIME OBSERVATION PREDICTION RESIDUAL RMSE_NSQ MAE_NSQ BIAS_NSQ RMSE_NSQ_OVER_MISSING_VALUE_ONLY MAE_NSQ_OVER_MISSING_VALUE_ONLY BIAS_NSQ_OVER_MISSING_VALUE_ONLY ts_k KEEP_missingdataFlag
                
                %RMSE_NSQ = sqrt(mean((KEEP_MOD - TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA).^2))
                %MAD_NSQ = mean(abs(KEEP_MOD - TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA))
                
                %                 if ACTIVATE_MISSINGSAMPLE == 0
                %
                %                     thermalsunrise = sr(1);
                %
                %
                %                     [XLS_ROW_NUMBER_POTENTIAL FILE_NUMBER_POTENTIAL] = resultReport2(FILENAMEPART, ROW, COLUMN, TEST_FIRE_CYCLE_NUMBER, TEST_CYCLE_TIME, RESIDUAL, FCTCOUNT_A, ITERATIONS_A, thermalsunrise, ts_M, tm_M, RMSE_NSQ, MAE_NSQ, BIAS_NSQ, XLS_ROW_NUMBER_POTENTIAL,FILE_NUMBER_POTENTIAL,EXCEL,Excel_Potential);
                %
                %
                %                 else
                %                     [XLS_ROW_NUMBER_POTENTIAL FILE_NUMBER_POTENTIAL] = resultReport4(FILENAMEPART, ROW, COLUMN, TEST_FIRE_CYCLE_NUMBER, GAP, RMSE_NSQ, MAE_NSQ, BIAS_NSQ, RMSE_NSQ_OVER_MISSING_VALUE_ONLY, MAE_NSQ_OVER_MISSING_VALUE_ONLY, BIAS_NSQ_OVER_MISSING_VALUE_ONLY,XLS_ROW_NUMBER_POTENTIAL,FILE_NUMBER_POTENTIAL,EXCEL,Excel_Potential);
                %
                %                 end
                %
                %
                %
                %                 XLS_ROW_NUMBER_POTENTIAL = XLS_ROW_NUMBER_POTENTIAL +1;
                %
                
                
                [TEST_FIRE_CYCLE_NUMBER ROW COLUMN]
                
                clear TEST_FIRE_CYCLE w TEST_CYCLE_TIME latitude longitude column row To_M  Ta_M tm_M ts_M dT_M Q R1 bA MNs mp cycle_out_scope
                %WATER BODY MASK
                %CLOUD MASK
                %REPORT UNDECIDED BASED ON cycle_out_scope (create another function)
                %FIRE MASK BEFORE MODELLING
            end
        end
        
    end
    
    % if NONSEQUENTIAL ==0
    %     clear FNDATAREGION
    %     save C:\Users\juwilingiye\UJ_PREDICTION1\TESTINGCYCLE_DATA TESTINGCYCLE_DATA
    % end
    
    
    %      figure
    %
    % %plot(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_TIME,KEEP_MOD,'r')
    % plot(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_TIME,KEEP_MOD,'r')
    % hold on
    % %plot(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_TIME, TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA,'b')
    % plot(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_TIME(KEEP_missingdataFlag==0),TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA(KEEP_missingdataFlag==0),'bx')
    % plot(TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_TIME(KEEP_missingdataFlag==1),TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA(KEEP_missingdataFlag==1),'gs')
    % hold off
    % RESIDUAL_NSQ = KEEP_MOD - TESTINGCYCLE_DATA(INDEX_ROW,INDEX_COLUMN).TEST_CYCLE_DATA;
    % RMSE_NSQ = sqrt(mean(RESIDUAL_NSQ.^2));
    % MAE_NSQ = mean(abs(RESIDUAL_NSQ));
    % BIAS_NSQ = sum(RESIDUAL_NSQ);
    %
    % [RMSE_NSQ MAE_NSQ BIAS_NSQ]
    %
    % figure
    % plot(RESIDUAL)
    
    
    %else
    
    %     switch ERRFUNCTION
    %         case 'L',
    %             load C:\Users\juwilingiye\UJ_PREDICTION1\vandenBergh_2006_LS\TESTINGCYCLE_DATA
    %         case 'C',
    %             load C:\Users\juwilingiye\UJ_PREDICTION1\vandenBergh_2006_Chi\TESTINGCYCLE_DATA
    %         case 'R',
    %             load C:\Users\juwilingiye\UJ_PREDICTION1\vandenBergh_2006_Rob\TESTINGCYCLE_DATA
    %     end
    
    
    disp('sequential started')
    
    %Check the length of time is equal for all pixels
    CHECK_LENGTH = [];
    [NUMBER_ROW NUMBER_COLUMN] = size(TESTINGCYCLE_DATA);
    for iN= 1:NUMBER_COLUMN
        for jN = 1:NUMBER_ROW
            CHECK_LENGTH = [CHECK_LENGTH length(TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME)]; %CAN BE DISPLAYED ON COMMAND WINDOW TO CHECK NUMBER OF SAMPLES FOR EACH PIXEL
        end
    end
    
    if(sum(CHECK_LENGTH == CHECK_LENGTH(1))==NUMBER_COLUMN*NUMBER_ROW)
        display('time length is the same for all pixels');
    end
    
    %Check the first time stamp
    CHECK_TIME = [];
    for iN= 1:NUMBER_COLUMN
        for jN = 1:NUMBER_ROW
            CHECK_TIME(jN,iN) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME(1);
        end
    end
    
    [iCT jCT] = max(CHECK_TIME);
    
    for iN= 1:NUMBER_COLUMN
        for jN = 1:NUMBER_ROW
            TESTINGCYCLE_DATA(jN,iN).START_INDEX = find(TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME(1:48)==iCT(1));
        end
    end
    
    
    
    
    NNNN_ = NTC
    %RRRR_ = RIGHT_CYCLE
    %IIII_
    %KeepINDEXROWCOLUM