function PH = backgroundValueOfObservation(EnsPX_,HX)

PH = [];
Ne = size(EnsPX_,3);
% size(EnsPX_)
% size(HX)
for pixel = 1:size(EnsPX_,2)
%     SEnsPX_HX = zeros(size(EnsPX_,1),size(HX,1));
%     for member = 1:Ne
%         SEnsPX_HX = SEnsPX_HX + EnsPX_(:,pixel,member)* (HX(:,pixel,member))';
%     end
    SEnsPX_HX = squeeze(EnsPX_(:,pixel,:))*(squeeze(HX(:,pixel,:)))';
    PH = [PH SEnsPX_HX/(Ne-1)]; %6 x (2.Pixels)
end