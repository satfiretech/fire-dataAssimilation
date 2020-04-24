function PIX = geocoordtopixcoord(GEO)
% clear all
% clc

%Convert from geo coordinate(latitude, longitude) to pixel coordinate(line, column)
%R. Wolf, (1999), 'Coordination Group for Meteorological Satellites LRIT/HRIT
%Global Specification', File: CGMS 03 – LRIT HRIT Global Spec. 2.6

%latitude = GEO(:,1) and longitude = GEO(:,2)
%line = PIX(:,1) and column = PIX(:,2)


% latitude = -25.245;
% longitude = 30.76;

latitude = GEO(:,1);
longitude = GEO(:,2);



NL = 464;
k = 1; %k range from 1 to 8 , segment 1 correspond to most South, NL is the number of line in one image segment file
COFF = 1856;
LOFF = 1856-(k-1)*NL;%(depend of the segment number) 1856; 
CFAC = -781648343;
LFAC = -781648343;



for i=1:length(latitude)
    lat = latitude(i) * pi/180;
    lon = longitude(i) * pi/180;

    c_lat = atan(0.993243*tan(lat));

    rl = 6356.5838/sqrt(1 - 0.00675701*cos(c_lat)^2);
    r1 = 42164 - rl*cos(c_lat)*cos(lon);
    r2 = -rl * cos(c_lat)*sin(lon);
    r3 = rl*sin(c_lat);
    rn = sqrt(r1^2 + r2^2 + r3^2);

    xx = atan(-r2/r1);
    yy = asin(-r3/rn);
    cc = COFF + xx*2^(-16)*CFAC;
    ll = LOFF + yy*2^(-16)*LFAC;

    ccd = cc - floor(cc);
    lld = ll - floor(ll);
    
    if ccd > 0.5
        column(i) = ceil(cc);
    else
        column(i) = floor(cc);
    end
    
    if lld > 0.5
        line(i) = ceil(ll); %line=row
    else
        line(i) = floor(ll);
    end
    % line = row
    % column

end

PIX = [line' column'];