%This file is mostly the same as the file c:/c:\MATLAB71\work\FireDetection2\COMPARE_FIR_MODIS\check4
close all
clear all
clc


%addpath('DeletedFile')

addpath('c:\MATLAB71\work\NNDeconvolution')  %To get water mask
addpath('c:\MATLAB71\work\FireDetection2\COMPARE_FIR_MODIS') % To get FIRfileCorrected and MODISfile

load MODISfile
load FIRfileCorrected
%load FIRfile

%load waterMask
load waterMask3(use this)%This is a change to this file

%compareFIR_MODIS.m must be run first


%To find out those pixels that were not on fire according to MOD14 or FIR
%from 30-07-2007 up to 05-08-2007 for the region considered
for i=1:size(MODIS,1)
    for j=1:size(MODIS,2)
        if length(MODIS(i,j).Latitude)==0
            MatrixMODIS(i,j) = 0;
        else
            MatrixMODIS(i,j) = 1;
        end
    end
end

for i=1:size(FIR,1)
    for j=1:size(FIR,2)
        if length(FIR(i,j).Latitude)==0
            MatrixFIR(i,j) = 0;
        else
            MatrixFIR(i,j) = 1;
        end
    end
end

PIXELAFFECTED = MatrixMODIS | MatrixFIR; %MatrixFIR; %MatrixMODIS | MatrixFIR;
WATERPIXELS = ~waterMask;

PIXELREMOVED = PIXELAFFECTED | WATERPIXELS;

% figure
% imagesc(PIXELAFFECTED)
% 
% figure
% imagesc(PIXELREMOVED)


[Ri Rj] = find(PIXELREMOVED==0); %Sample from pixels not affected with fire and not falling in water body.

rand('twister',5489)

CHOOSENPIXEL = ceil(length(Ri)*rand(2505,1)); %Simple random sampling


SAMPLEPIXELS = zeros(2505,2);
for i = 1:2505
    SAMPLEPIXELS(i,:) = [Ri(CHOOSENPIXEL(i)) Rj(CHOOSENPIXEL(i))];
end

SAMPLEPIXELS(1,:)


%1st cycle

S1 = ceil(96*rand(2500,4));

%2nd cycle

S2 =ceil(96*rand(2500,4)) + 96;

%3rd cycle

S3 =ceil(96*rand(2500,4)) + 96*2;

%4th cycle

S4 =ceil(96*rand(2500,4)) + 96*3;

%5th cycle

S5 =ceil(96*rand(2500,4)) + 96*4;

TIMESTAMPSAMPLED = [S1 S2 S3 S4 S5];

%save MonteCarloRuns SAMPLEPIXELS TIMESTAMPSAMPLED

PLOTSAMPLEPIXELS = zeros(size(PIXELREMOVED));
for i=1:2500
    PLOTSAMPLEPIXELS(SAMPLEPIXELS(i,1),SAMPLEPIXELS(i,2)) = 1;
end

imagesc(PLOTSAMPLEPIXELS)

%I WILL MULTIPLY IT BY 4. INSTEAD OF 2500 IT WILL BE 10,000 PIXELS
    

% geocoordtopixcoord([-25.3584667 31.8934972;
% -25.4618694 31.532972;
% -25.15535 31.1975;
% -25.0251167 31.2414306;
% -24.981306 31.4852556;
% -24.4758194 31.390775;
% -23.9454528 31.1650417;
% -22.7383028 31.0090333;
% -22.400167 31.0414111]); %Pixels which make boarder of the Kruger National Park according to Wikipedia
% 
% %Pixels falling in Kruger National Park which are affected by fire, from
% %31-07-2007 to 04-08-2007
% [J K]= find(PIXELAFFECTED==1);
% r = 0;
% for m = 1:length(J)
%     if (K(m)>=57)&(K(m)<=76)&(J(m)>=195)&(J(m)<=319)
%         r = r + 1;
%         KEEP(r,:) = [J(m) K(m)];
%     end
% end
