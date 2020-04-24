function THRESHOLD_POTENTIAL = getThresholdPotential(KeepScaledResidual,timestamp,THRESHOLD_COEFFICIENT_POTENTIAL)




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

%tic
for t = 1:96
    for i = 1:size(KeepScaledResidual,1)
        for j = 1:size(KeepScaledResidual,2)
            %             timestamp-k(j,i)
            %             A = ceil(1/2*(k(j,i) + timestamp - 1)/(timestamp - 1));
            %             B = ceil(k(j,i)/(timestamp - 1));
            %             timestamp - 2*(mod(B,2)-0.5)*(k(j,i) - 2*(A-1)*(timestamp - 1))

            %SAMPLE = timestamp-k(j,i);
            SAMPLE = timestamp-k(i,j);
            if SAMPLE<1 | SAMPLE>(timestamp-1)
                %S1 = ceil(1/2*(k(j,i) + timestamp - 1)/(timestamp - 1));
                %S2 = ceil(k(j,i)/(timestamp - 1));
                %SAMPLE = timestamp - 2*(mod(S2,2)-0.5)*(k(j,i) - 2*(S1-1)*(timestamp - 1));
                S1 = ceil(1/2*(k(i,j) + timestamp - 1)/(timestamp - 1));
                S2 = ceil(k(i,j)/(timestamp - 1));
                SAMPLE = timestamp - 2*(mod(S2,2)-0.5)*(k(i,j) - 2*(S1-1)*(timestamp - 1));
            end
            %
            while isnan(KeepScaledResidual(i,j,SAMPLE))==1 | isinf(KeepScaledResidual(i,j,SAMPLE))==1 %condition on k(j) so that it does not exceed the limit, also insert (active) fire detection results
                k(i,j) = k(i,j) + 1;
                %SAMPLE = timestamp-k(j,i);
                SAMPLE = timestamp-k(i,j);
                if SAMPLE<1 | SAMPLE>(timestamp-1)
                    %S1 = ceil(1/2*(k(j,i) + timestamp - 1)/(timestamp - 1));
                    %S2 = ceil(k(j,i)/(timestamp - 1));
                    %SAMPLE = timestamp - 2*(mod(S2,2)-0.5)*(k(j,i) - 2*(S1-1)*(timestamp - 1));
                    S1 = ceil(1/2*(k(i,j) + timestamp - 1)/(timestamp - 1));
                    S2 = ceil(k(i,j)/(timestamp - 1));
                    SAMPLE = timestamp - 2*(mod(S2,2)-0.5)*(k(i,j) - 2*(S1-1)*(timestamp - 1));
                end
            end
            %previousScaledResidual(i,j,t) = KeepScaledResidual(i,j,timestamp-k(i,j));
            previousScaledResidual(i,j,t) = KeepScaledResidual(i,j,SAMPLE);
            k(i,j) = k(i,j) + 1;
        end
    end
end
%toc


load gumbelCoefficients; %Take it on top

parameters = zeros(size(KeepScaledResidual,1),size(KeepScaledResidual,2),2);
for i = 1:size(KeepScaledResidual,1)
    for j = 1:size(KeepScaledResidual,2)
        pSR = squeeze(previousScaledResidual(i,j,:));
        parameters(i,j,1:2) = gumbelCoefficients* pSR;
    end
end


% length(find(isnan(C)==1))
% length(find(isinf(C)==1))





%[lG sG]= gumbelCoefficients*previousScaledResidual;
THRESHOLD_POTENTIAL = parameters(:,:,1) + THRESHOLD_COEFFICIENT_POTENTIAL*parameters(:,:,2);
% disp('START')
% disp('start')
% parameters(:).'
% [mean(pSR(:)) sqrt(6*var(pSR(:)))/pi]
% [skewness(pSR(:)) kurtosis(pSR(:))] 
% disp('end')
% 
% 
AA = parameters(:,:,1);
BB = parameters(:,:,2);

if BB<0
%figure
%hist(pSR)

%CC = sprintf('[%d %d] & [%d %d]',AA, BB, mean(pSR(:)) ,sqrt(6*var(pSR(:)))/pi, skewness(pSR(:)), kurtosis(pSR(:)));
%title(CC)
disp('POSITIVE SKEWNESS')
end