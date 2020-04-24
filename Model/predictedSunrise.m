function iPredictedSunrise =   predictedSunrise(TIME, DATE, iSunrise, longitude, latitude)
%Predicted from cycle 8 up to the end of all cycles.


for iday = 1:size(DATE,1)
    year = DATE(iday,1);
    month = DATE(iday,2);
    day = DATE(iday,3);
    [sunrise90 sunrise90 solarnoon90 sunset90 daylength90] = solartime2([90 90 90],year,month,day,longitude,latitude);
    daySunrise90(iday) = sunrise90;
end

%Change this time so that 23-07-2007 00:00 be the 0 hour. The time
%is expressed in hours.
morningHorizonTime = zeros(1,size(DATE,1));
for iday = 1:size(DATE,1)
    morningHorizonTime(iday) = (iday-1)*24 + daySunrise90(iday);
end


SunriseLag = -morningHorizonTime + TIME(iSunrise); %To be changed.
SUNRISE = TIME(iSunrise);

for i = 8:size(DATE,1)
    SUNRISE(i) =  morningHorizonTime(i) + mean(SunriseLag(i-7:i-1));
end

%iPredictedSunrise = zeros(1,size(DATE,1));
iPredictedSunrise = iSunrise;

%Find the sunrise point samples
for iday = 8:size(DATE,1)
    iBeforeSunrise = find(TIME<=SUNRISE(iday));
    sBeforeSunrise = [iBeforeSunrise.' TIME(iBeforeSunrise).'];
    tBeforeSunrise  = sBeforeSunrise(:,2);
    [dfBeforeSunrise ifBeforeSunrise] = max(tBeforeSunrise);
    %Prediction with the highest among the lower times (floor operation)
    iPredictedSunrise(iday) = sBeforeSunrise(ifBeforeSunrise,1);
    
    %Use of a closer sample
    %if (SUNRISE(iday) - TIME(iPredictedSunrise(iday))) > (TIME(iPredictedSunrise(iday)+1) - SUNRISE(iday))
    %    iPredictedSunrise(iday) = iPredictedSunrise(iday) + 1;
    %end
end

%No prediction
%iPredictedSunrise = iSunrise;

%[iSunrise;
% iPredictedSunrise]   
