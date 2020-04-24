function [FIREFLAG FEATURE y_predicted] = startTimeFireDetect(y, Keep_Q2, x_, tkcurrent, threshold, RESIDUAL)


%     SLAF to get for 4DVar
%     x_
%     filtered is x_


NUMBER_ENSEMBLE_MEMBERS = size(x_,3);
forecastSteps = 1;

y_extended = extend(y,NUMBER_ENSEMBLE_MEMBERS);
y_predicted = observationFunction(x_,y_extended,tkcurrent); %CORRECT y_extended
y_predictedkeep(forecastSteps,1:size(y_predicted,1)*size(y_predicted,2),1:size(y_predicted,3)) = reshape(y_predicted,1,size(y_predicted,1)*size(y_predicted,2),size(y_predicted,3));
ypredictedMean = mean(y_predictedkeep,3); %Ensemble
y_predicted = reshape(ypredictedMean(1,:),size(y,1),size(y,2));



if RESIDUAL == 0 %Prediction Residual
    %threshold = 2;
    [FIREFLAG PR] = predictionResidual(y, y_predicted,threshold);
    %KeepFireDetect(:,:,timestamp+le) = FIREFLAG;
    %PRkeep(:,:,timestamp+le) = PR;
    FEATURE = PR;

elseif RESIDUAL == 1  %OL
    %threshold = 2;
    %[FIREFLAG OL] = OLResidual(y, Keep_Q2, y_predictedkeep, y_predicted, threshold);
    [FIREFLAG OL] = ssrOLResidual(y, Keep_Q2, y_predictedkeep, y_predicted, threshold);
    %KeepFireDetect(:,:,timestamp+le) = FIREFLAG;
    %OLkeep(:,:,timestamp+le) = OL;
    FEATURE = OL;

elseif RESIDUAL == 2  %gELL
    %threshold = 0.2;
    %[FIREFLAG gELL] = ELLResidual(y, Keep_Q2, y_predictedkeep, y_predicted, x_, tkcurrent, threshold);
    [FIREFLAG gELL] = ssrELLResidual(y, Keep_Q2, y_predictedkeep, y_predicted, x_, tkcurrent, threshold);
    %KeepFireDetect(:,:,timestamp+le) = FIREFLAG;
    %gELLkeep(:,:,timestamp+le) = gELL;
    FEATURE = gELL;

else
    display('Specify residual (feature)')
end


    