function THRESHOLD_POTENTIAL = getThresholdPotential(KeepScaledResidual,timestamp,THRESHOLD_COEFFICIENT_POTENTIAL,LATEST_THRESHOLD_POTENTIAL)



% clear all
% 
% clc
% 
% 
% A = randn(5,200);
% 
% B = randperm(500);
% 
% A(B(1:100)) = NaN;
% A(B(110:200)) = Inf;
% 
% length(find(isnan(A)==1))
% length(find(isinf(A)==1))

%k = ones(size(KeepScaledResidual,1)*size(KeepScaledResidual,2),1);
k = ones(size(KeepScaledResidual,1),size(KeepScaledResidual,2)); 
previousScaledResidual = zeros(size(KeepScaledResidual,1),size(KeepScaledResidual,2),96);
FLAG_KEEP_LAST = zeros(size(KeepScaledResidual,1),size(KeepScaledResidual,2)); 
%tic
for t = 1:96
    for i = 1:size(KeepScaledResidual,1)
        for j = 1:size(KeepScaledResidual,2)
            SAMPLE = timestamp-k(i,j);
            if SAMPLE >=1
            while (isnan(KeepScaledResidual(i,j,SAMPLE))==1 | isinf(KeepScaledResidual(i,j,SAMPLE))==1) %condition on k(j) so that it does not exceed the limit, also insert (active) fire detection results
                k(i,j) = k(i,j) + 1;
                SAMPLE = timestamp-k(i,j);
                if SAMPLE < 1
                    break;
                end
            end
            end
            if SAMPLE < 1
                %THRESHOLD_POTENTIAL(i,j) = LATEST_THRESHOLD_POTENTIAL(i,j); 
                FLAG_KEEP_LAST(i,j) = 1;
                %Number of samples to define distribution is less than 96
            else
                previousScaledResidual(i,j,t) = KeepScaledResidual(i,j,SAMPLE);
                k(i,j) = k(i,j) + 1;
            end
        end
    end
end
%toc


load gumbelCoefficients; %Take it on top

parameters = zeros(size(KeepScaledResidual,1),size(KeepScaledResidual,2),2);
for i = 1:size(KeepScaledResidual,1)
    for j = 1:size(KeepScaledResidual,2)
        if FLAG_KEEP_LAST(i,j)~=1 %If 96 samples are available
            pSR = squeeze(previousScaledResidual(i,j,:));
            parameters(i,j,1:2) = gumbelCoefficients* sort(pSR(:));
            KEEP_PARAMETERS = parameters(i,j,:);
            KEEP_PARAMETERS = KEEP_PARAMETERS(:);
            if KEEP_PARAMETERS(2)<0 %If scale parameter is negative
                FLAG_KEEP_LAST(i,j)=1;
                parameters(i,j,1:2) = [0;0];
            end
        end
    end
end


% length(find(isnan(C)==1))
% length(find(isinf(C)==1))





%[lG sG]= gumbelCoefficients*previousScaledResidual;
THRESHOLD_POTENTIAL = parameters(:,:,1) + THRESHOLD_COEFFICIENT_POTENTIAL*parameters(:,:,2);

THRESHOLD_POTENTIAL = THRESHOLD_POTENTIAL + FLAG_KEEP_LAST.*LATEST_THRESHOLD_POTENTIAL; 
                

% disp('START')
% disp('start')
% parameters(:).'
% [mean(pSR(:)) sqrt(6*var(pSR(:)))/pi]
% [skewness(pSR(:)) kurtosis(pSR(:))] 
% disp('end')
% 
% 

%AA = parameters(:,:,1);
%BB = parameters(:,:,2);

%[AA BB THRESHOLD_POTENTIAL]
%if BB<0
%figure


% % % % % % % % if FLAG_KEEP_LAST==0
% % % % % % % % 
% % % % % % % % 
% % % % % % % % %[AA BB -sign(skewness(pSR))*sqrt(6*var(pSR))/pi*0.5772156649015328606 sqrt(6*var(pSR))/pi]
% % % % % % % %     
% % % % % % % %     figure
% % % % % % % %     
% % % % % % % %     [cdfF valueX] = ecdf(pSR);
% % % % % % % % Bin_.rule = 1;
% % % % % % % % [C_,E] = dfswitchyard('dfhistbins',pSR,[],[],Bin_,cdfF,valueX); %Freedman-Diaconis rule
% % % % % % % % [C N] = ecdfhist(cdfF,valueX,'edges',E);
% % % % % % % % %figure(FIG_PDF);
% % % % % % % % h = bar(N,C,'hist');
% % % % % % % % set(h,'FaceColor','none','EdgeColor',[0.333333 0 0.666667],'LineStyle','-', 'LineWidth',1);
% % % % % % % % 
% % % % % % % % 
% % % % % % % % MIN = min(pSR);
% % % % % % % % MAX = max(pSR);
% % % % % % % % ABSCISS = MIN:0.01:MAX; %or ABSCISS = C or X;
% % % % % % % % hold on
% % % % % % % % 
% % % % % % % % 
% % % % % % % % 
% % % % % % % % SKEW_GUMBEL = sign(skewness(pSR));
% % % % % % % % 
% % % % % % % % %BB = sqrt(6*var(pSR))/pi;
% % % % % % % % %AA = - SKEW_GUMBEL * BB*0.5772156649015328606;
% % % % % % % % 
% % % % % % % % GUMBEL_DATA = exp(-SKEW_GUMBEL*(ABSCISS-AA)/BB).*exp(-exp(-SKEW_GUMBEL*(ABSCISS-AA)/BB))/BB;
% % % % % % % % plot(ABSCISS,GUMBEL_DATA,'k');
% % % % % % % % 
% % % % % % % % hold off
% % % % % % % % 
% % % % % % % % title(sprintf('[%d %d %d]',AA,BB,THRESHOLD_POTENTIAL))
% % % % % % % % 
% end

%CC = sprintf('[%d %d] & [%d %d]',AA, BB, mean(pSR(:)) ,sqrt(6*var(pSR(:)))/pi, skewness(pSR(:)), kurtosis(pSR(:)));
%title(CC)
%disp('POSITIVE SKEWNESS')
%end