function [latitude, longitude] = pixcoordtogeocoord(row,column)
% clear all
% clc

%Convert from pixel coordinates (line, column) to geo coordinates (latitude, longitude)
%[latitude, longitude] = pixcoordtogeocoord(line,column)

% row = 995;
% column = 913;


COFF = 1856;
LOFF = 1856; %(depend of the segment number)
CFAC = -781648343;
LFAC = -781648343;


x = 2^16*(column-COFF)/CFAC;
y = 2^16*(row-COFF)/CFAC;

sd = sqrt((42164*cos(x)*cos(y))^2 - (cos(y)^2 + 1.006803*sin(y)^2)*1737121856);
sn = (42164*cos(x)*cos(y) - sd)/(cos(y)^2 + 1.006803*sin(y)^2);
s1 = 42164 - sn*cos(x)*cos(y);
s2 = sn*sin(x)*cos(y);
s3 = -sn*sin(y);
sxy = sqrt(s1^2 + s2^2);

lon = atan(s2/s1);
lat = atan(1.006803*s3/sxy);

latitude = lat*180/pi;
longitude = lon*180/pi;

% latitude
% longitude



