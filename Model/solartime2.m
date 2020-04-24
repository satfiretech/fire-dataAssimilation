function [SUNRISEL SUNRISER SOLARNOON SUNSET DAYLENGTH] = solartime2(zenith,year,month,day,longitude,latitude) %SOLAR NOON, SUNRISE, SUNSET, DAY LENGTH

%[sunrise solarnoon sunset daylength dayofyear] = solartime(year,month,day,longitude,latitude)
%Return UTC apparent sunset, apparent sunrise and solar noon in hours and
%Equation of time is used to calculate solar noon.
%INPUT
%latitude: Positive for NORTH and Negative for SOUTH (location for sunrise/sunset)
%longitude: Positive for EAST and Negative for WEST (location for sunrise/sunset)
%year:xxx (numerical) Year of sunrise/sunset
%month:(x)x (numerical)  Month of sunrise/sunset 
%day: Day of the month (numerical) Date of the month for sunrise/sunset (day,month,year:date of sunrise/sunset)
%OUTPUT in solar UTC
%sunrise (hour)
%solarnoon (hour)
%sunset (hour)
%daylength (hour)
%Results are in terms of hour, but to know minutes, the decimal part must
%be multiplied by 60, and again the decimal parts of minutes multiplied by
%60 to know seconds for time displayed as xx:xx:xx.


%First calculate the day of the year
N1 = floor(275 * month / 9);
N2 = floor((month + 9) / 12);
N3 = (1 + floor((year - 4 * floor(year / 4) + 2) / 3));
N = N1 - (N2 * N3) + day - 30;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculate EQUATION OF TIME and it is expressed in minutes
%---------------------------------------------------------
PERIHELION = [2007 3
              2008 3
              2009 4
              2010 3
              2011 3
              2012 5
              2013 2
              2014 4
              2015 4
              2016 2
              2017 4
              2018 3
              2019 3
              2020 5];

          
M = 2*pi*(N - PERIHELION(PERIHELION==year,2))/365.242;   %N - 3: Time from periapsis (in 2007 it was 03/Jan/2007)
EQTIME = -7.657*sin(M) + 9.862*sin(2*M + 3.599); %Wikipedia NEW AND SIMILAR TO Whitman_2003

% M = 2*pi*(N-3)/365.24;
% EQTIME2 = 229.18*(-0.0334*sin(M) + 0.04184*sin(2*M + 3.5884)); %Whitman_2003 and similar to Wikipedia NEW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculate SUNRISE and SUNSET and there are expressed in hours
%-------------------------------------------------------------
% Source:
% Almanac for Computers, 1999
% published by Nautical Almanac Office
% United States Naval Observatory
% Washington, DC 20392

localOffset = 0;
%zenith = 78; %Sun's zenith for sunrise/sunset, official solar zenith at sunrise and sunrise (other options are: official=90 degrees 50', civil=96 degrees, nautical=102 degrees and astronomical = 108 degrees)

%90 + 50/60
%96 
%102 
%108 


%Convert the longitude to hour value and calculate an approximate time
lngHour = longitude / 15;
%if rising time is desired:
t_rising = N + ((6 - lngHour) / 24);
%if setting time is desired:
t_setting = N + ((18 - lngHour) / 24);

%calculate the Sun's mean anomaly
M_rising = (0.9856 * t_rising) - 3.289;
M_setting = (0.9856 * t_setting) - 3.289;

%calculate the Sun's true longitude (Ecliptic longitude)
L_rising = M_rising + (1.916 * sind(M_rising)) + (0.020 * sind(2 * M_rising)) + 282.634;
L_rising = mod(L_rising,360);
L_setting = M_setting + (1.916 * sind(M_setting)) + (0.020 * sind(2 * M_setting)) + 282.634;
L_setting = mod(L_setting,360);
%NOTE: L potentially needs to be adjusted into the range [0,360) by adding/subtracting 360

%calculate the Sun's right ascension
RA_rising = atand(0.91764 * tand(L_rising));
RA_rising = mod(RA_rising,360);
RA_setting = atand(0.91764 * tand(L_setting));
RA_setting = mod(RA_setting,360);
%NOTE: RA potentially needs to be adjusted into the range [0,360) by adding/subtracting 360

%right ascension value needs to be in the same quadrant as L
Lquadrant_rising  = (floor( L_rising/90)) * 90;
RAquadrant_rising = (floor(RA_rising/90)) * 90;
RA_rising = RA_rising + (Lquadrant_rising - RAquadrant_rising);

Lquadrant_setting  = (floor( L_setting/90)) * 90;
RAquadrant_setting = (floor(RA_setting/90)) * 90;
RA_setting = RA_setting + (Lquadrant_setting - RAquadrant_setting);

%right ascension value needs to be converted into hours
RA_rising = RA_rising / 15;
RA_setting = RA_setting / 15;

%calculate the Sun's declination
sindDec_rising = 0.39782 * sind(L_rising);
cosdDec_rising = cosd(asind(sindDec_rising));

sindDec_setting = 0.39782 * sind(L_setting);
cosdDec_setting = cosd(asind(sindDec_setting));

%calculate the Sun's local hour angle
cosdH_rising = (cosd(zenith) - (sindDec_rising * sind(latitude))) / (cosdDec_rising * cosd(latitude));

if (cosdH_rising >  1)
    disp('the sun never rises on this location (on the specified date)');%the sun never rises on this location (on the specified date)
    return;
end
if (cosdH_rising < -1)
    disp('the sun never sets on this location (on the specified date)');%the sun never sets on this location (on the specified date)
    return;
end

cosdH_setting = (cosd(zenith) - (sindDec_setting * sind(latitude))) / (cosdDec_setting * cosd(latitude));
if (cosdH_setting >  1)
    disp('the sun never rises on this location (on the specified date)');%the sun never rises on this location (on the specified date)
    return;
end
if (cosdH_setting < -1)
    disp('the sun never sets on this location (on the specified date)');%the sun never sets on this location (on the specified date)
    return;
end


%finish calculating H and convert into hours
%if rising time is desired:
H_rising = 360 - acosd(cosdH_rising);
%if setting time is desired:
H_setting = acosd(cosdH_setting);

H_rising = H_rising / 15;
H_setting = H_setting / 15;

%calculate local mean time of rising/setting
T_rising = H_rising + RA_rising - (0.06571 * t_rising) - 6.622;

T_setting = H_setting + RA_setting - (0.06571 * t_setting) - 6.622;

%adjust back to UTC
UT_rising = T_rising - lngHour;
UT_rising = mod(UT_rising,24);
UT_setting = T_setting - lngHour;
UT_setting = mod(UT_setting,24);
%NOTE: UT potentially needs to be adjusted into the range [0,24) by adding/subtracting 24

%convert UT value to local time zone of latitude/longitude
localT_rising = UT_rising + localOffset;
localT_setting = UT_setting + localOffset;

SUNRISE = UT_rising;%-EQTIME/60; %Apparent sunrise
SUNSET = UT_setting;%-EQTIME/60; %Apparent sunset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculate SOLAR NOON
%--------------------

SOLARNOON = (12*60 - longitude/15*60 - EQTIME)/60;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculate DAY LENGTH
%--------------------
%Use Brock model (Forsythe_1995)
%Earth declination angle (in degrees) as a function of day of year
Dec = 23.45*sind(360*(283+N)/365);  %Calculation of the declination to the sun of a given pixel at a given day of the year

% Dec = 23.45*sind(360*(284+N)/365.25);  %(sun3_)
% Dec = 23.5*sind(360*(284+N)/365);      %Kaufmann and Weatherred (1982)

%sunrise/sunset hour angle with northern latitude being positive (latitude NOrth is positive)
H = acosd(-tand(latitude)*tand(Dec));  %1)Taken altitude angle = 0 at sunset and sunrise, no correction is applied
% 2)Must implement for location where the sunrise and sunset does not occur
% If a surface is tilted from the horizontal the Sun may rise over its edge after it has rise over the 
% horizon. Therefore the surface may shade itself for some of the day. The sunrise and sunset angles
% for a titled surface facing the equator (i.e. facing due south in the
% northern hemisphere) are given by: (in this case the angle inclination of
% the surface from the horizontal is taken  into account) (sun3_.pdf)


%DAYLENGTH = 2*H/15;   %Calculation of w (assumed equals to the hours of daylight) 
DAYLENGTH = UT_setting - UT_rising;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DAYLENGTH: ANOTHER APPROACH (http://herbert.gandraxa.com/herbert/lod.asp)
% m = 1 - tand(latitude)*tand(23.439*cos(0.0172*211));
% DAYLENGTH = acosd(1-m)/180*24;

%DAYLENGTH + BOTH TWILIGHT
% n = 1 - tand(latitude)*tand(23.439*cos(0.0172*211)) + tand(12)/cos(latitude);
% DAYLENGHT_PLUS_TWILIGHT  = acosd(1-n)/180*24;


SUNRISEL = SUNRISE(1); 
SUNRISER = SUNRISE(3); 
%SOLARNOON 
SUNSET_A = SUNSET(3);
SUNSET = SUNSET_A;
DAYLENGTH_A = DAYLENGTH(2);
DAYLENGTH = DAYLENGTH_A;
