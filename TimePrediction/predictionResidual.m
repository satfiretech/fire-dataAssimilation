function [FIREFLAG r] = predictionResidual(ynewkeep, y_predicted,threshold)
%Prediction residual

r = ynewkeep - y_predicted;

FIREFLAG = zeros(size(r));

FIREFLAG(r>threshold) = 1;


% figure
% 
% %plot(squeeze(ynewkeep),'b*');hold on;plot(squeeze(mean(y_predictedkeep(:,pixel,:),3)),'r');hold off
% plot(squeeze(ynewkeep),'b*');hold on;plot(squeeze(mean(y_predictedkeep,3)),'r');hold off
% 
% 
% figure
% 
% plot(squeeze(r))
% 
% figure
% 
% plot(squeeze(FIREFLAG),'r')