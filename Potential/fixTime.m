function  [tkcurrentkeep y_predictedkeep SHIFT] = fixTime(TESTINGCYCLE_DATA,n,PREDICTION_INTERVAL,filtered,y,NUMBER_ENSEMBLE_MEMBERS)
%,ThermalSunrise,tHorizon)


NUMBER_ROW = size(TESTINGCYCLE_DATA,1);
NUMBER_COLUMN = size(TESTINGCYCLE_DATA,2);
SHIFT = []; %shift at start of each cycle
%SHIFT = zeros(NUMBER_ROW,NUMBER_COLUMN,1);


for forecastSteps = 1:PREDICTION_INTERVAL %+1
    Ft = 1;
    xp = Ft^(forecastSteps-1)* filtered; %xp or xsmooth

    for jN = 1:NUMBER_ROW
        for iN = 1:NUMBER_COLUMN
            nnew = n + forecastSteps;
            %ynew(jN,iN) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_DATA(nnew(jN,iN));
            tcurrent(jN,iN) = TESTINGCYCLE_DATA(jN,iN).TEST_CYCLE_TIME(nnew(jN,iN));
            tHorizonnew(jN,iN) = TESTINGCYCLE_DATA(jN,iN).tmorningHorizon(nnew(jN,iN));
            observedThermalSunrise(jN,iN) = TESTINGCYCLE_DATA(jN,iN).tSunriseDay(nnew(jN,iN));
        end
    end
    %tcurrent = t+15/60*forecastSteps;
    tkcurrent(:,:,forecastSteps) = tcurrent; %+ 24 *(1+floor(forecastSteps/96));
    %tkcurrent(:,:,forecastSteps) = tcurrent;
    tHorizonnewkeep(:,:,forecastSteps) = tHorizonnew;
    observedThermalSunrisekeep(:,:,forecastSteps) = observedThermalSunrise;
end

%TP = tkcurrent;
y_extended = extend(y,NUMBER_ENSEMBLE_MEMBERS);


%for forecastSteps = 1:PREDICTION_INTERVAL


%nnew = n + forecastSteps;


for jN = 1:NUMBER_ROW
    for iN = 1:NUMBER_COLUMN

        k = (iN-1)*NUMBER_ROW+jN;
        nnew(jN,iN) = n(jN,iN) + 1;
        %new = n;
        FLAG = 0;
        cyclenumberMinus1 = 0;
        while nnew(jN,iN) - n(jN,iN) <= PREDICTION_INTERVAL
            sumC = 0;
            i = 0;
            while TESTINGCYCLE_DATA(jN,iN).LEFT_CYCLE_SAMPLE(nnew(jN,iN))~=0
                %sumC = sumC + thermalsunriseFunction(x(:,i,j-1),y(2,i),t(i),tHorizon(i));
                sumC = sumC + thermalsunriseFunction(xp(:,k,:),y_extended(jN,iN,:),tkcurrent(jN,iN,nnew(jN,iN)-n(jN,iN)),extend(tHorizonnewkeep(jN,iN,nnew(jN,iN)-n(jN,iN)),NUMBER_ENSEMBLE_MEMBERS));
                %c_extended = extend(c,Ne);
                %c_perturbed = c_extended + extend(sqrt(Q2c),Ne).*randn(size(c_extended));
                %c_predicted = thermalsunriseFunction(xp,y_extended,tkcurrent(:,:,nnew),tHorizonnewkeep(:,:,nnew));
                %alphac = c_perturbed - c_predicted; %50 x 50 x Ne

                i = i + 1;
                nnew(jN,iN) = nnew(jN,iN) + 1;
                if nnew(jN,iN) - n(jN,iN) > PREDICTION_INTERVAL
                    FLAG = 1;
                    break;
                end
            end
            if FLAG == 0
                cyclenumberMinus1 = cyclenumberMinus1 + 1;
                sumC = sumC + thermalsunriseFunction(xp(:,k,:),y_extended(jN,iN,:),tkcurrent(jN,iN,nnew(jN,iN)-n(jN,iN)),extend(tHorizonnewkeep(jN,iN,nnew(jN,iN)-n(jN,iN)),NUMBER_ENSEMBLE_MEMBERS));
                meanC = sumC/i;
                %shiftTime = meanC - (tThermalSunrise - tHorizon);
                %[Ensemble of c] shiftTime = meanC - (observedThermalSunrisekeep(jN,iN,nnew) - tHorizonnewkeep(jN,iN,nnew))*ones(size(meanC));
                %Instead of ensemble of c, the mean is found
                shiftTime = mean(meanC) - (observedThermalSunrisekeep(jN,iN,nnew(jN,iN)-n(jN,iN)) - tHorizonnewkeep(jN,iN,nnew(jN,iN)-n(jN,iN)));
                shiftSample = round(shiftTime/0.25);

                SHIFT(jN,iN,cyclenumberMinus1) = shiftSample; 
                %nnew - 1 corresponds to
                %TESTINGCYCLE_DATA(jN,iN).LEFT_CYCLE_SAMPLE(nnew)=0, Not it
                %is newnew
                %m = nnew - 1;
                m = nnew(jN,iN);
                if sign(shiftSample)<0
                    if m+shiftSample+1<=n(jN,iN)
                        tkcurrent(jN,iN,n(jN,iN)+1-n(jN,iN):m-n(jN,iN)) = tkcurrent(jN,iN,n(jN,iN)+1-n(jN,iN):m-n(jN,iN)) - 24;
                    else
                        tkcurrent(jN,iN,m+shiftSample+1-n(jN,iN):m-n(jN,iN)) = tkcurrent(jN,iN,m+shiftSample+1-n(jN,iN):m-n(jN,iN)) - 24; %At 0, i.e., 0 correspond to n
                    end
                    
                else
                    
                    if m+shiftSample-n(jN,iN) > PREDICTION_INTERVAL & m+1-n(jN,iN) <= PREDICTION_INTERVAL
                        tkcurrent(jN,iN,m+1-n(jN,iN):PREDICTION_INTERVAL) = tkcurrent(jN,iN,m+1-n(jN,iN):PREDICTION_INTERVAL) + 24; %At 0, i.e., 0 correspond to n
                    elseif m+shiftSample-n(jN,iN) <= PREDICTION_INTERVAL & m+1-n(jN,iN) <= PREDICTION_INTERVAL
                        tkcurrent(jN,iN,m+1-n(jN,iN):m+shiftSample-n(jN,iN)) = tkcurrent(jN,iN,m+1-n(jN,iN):m+shiftSample-n(jN,iN)) + 24; %At 0, i.e., 0 correspond to n
                    else
                        disp('m+1>PREDICTION_INTERVAL')
                    end
                end

                nnew(jN,iN) = nnew(jN,iN) + 1;
            end
        end
    end
end


%I don't have to update the TESTINGCYCLE_DATA.ttsr
%tkcurrent = fixTime()
%tThermalSunrise = [c] + tHorizon; %Row x Column x Day
%tkcurrent




for forecastSteps = 1:PREDICTION_INTERVAL

    %Determine xp per forecastSteps
    Ft = 1;
    xp = Ft^(forecastSteps-1)* filtered; %xp or xsmooth

    %y_predicted(forecastSteps,:,:) = ensembleforecast(mean(xp,3),y_extended(1),tkcurrent(:,:,forecastSteps)); %CORRECT y_extended
    %y_predicted = observationFunction(x_,y_extended,t);
    %size(xp)
    %size(y_extended)
    y_predicted = observationFunction(xp,y_extended,tkcurrent(:,:,forecastSteps)); %CORRECT y_extended
    y_predictedkeep(forecastSteps,1:size(y_predicted,1)*size(y_predicted,2),1:size(y_predicted,3)) = reshape(y_predicted,1,size(y_predicted,1)*size(y_predicted,2),size(y_predicted,3));
end %[PREDICTION_INTERVAL x Pixels x Ensemble members

%y_predictedkeep(forecasteps,:,:)
tkcurrentkeep = tkcurrent(:,:,1:PREDICTION_INTERVAL);

