function [THRESHOLD_ACTIVE1 THRESHOLD_ACTIVE2] = getThresholdsDiffScaledResidual(KeepScaledResidual,CENTREL,CENTRE,N1L,N2L,ON,timestamp,THRESHOLD_COEFFICIENT_CONTEXTUAL,LATEST_THRESHOLD_ACTIVE1, LATEST_THRESHOLD_ACTIVE2)



%Neighbour = [-1 -1;-1 0;-1 1;0 1;1 1;1 0;1 -1;0 -1];
%Linear_Neighbour = (Neighbour(:,2))*size(PotentialFireDetect,1) + Neighbour(:,1);


% ON1 = 1*((ones(8,1)*CENTREL +   Linear_Neighbour*ones(1,length(CENTREL)))>0 & (ones(8,1)*CENTREL +   Linear_Neighbour*ones(1,length(CENTREL)))<= size(PotentialFireDetect,1)*size(PotentialFireDetect,2));  % If on bound of the whole image the difference must be 0 (the image is extended by repeat)
% ON2 = 2*((ones(8,1)*CENTREL + 2*Linear_Neighbour*ones(1,length(CENTREL)))>0 & (ones(8,1)*CENTREL + 2*Linear_Neighbour*ones(1,length(CENTREL)))<= size(PotentialFireDetect,1)*size(PotentialFireDetect,2));
% ON2 = max(ON1,ON2);
%
%
%
% N1L = ones(8,1)*CENTREL + ON1.*(Linear_Neighbour*ones(1,length(CENTREL)));
% N2L = ones(8,1)*CENTREL + ON2.*(Linear_Neighbour*ones(1,length(CENTREL)));

%[N1L N2L]

% ones(8,1)*[3 4 7 10 2] - NEIGHBOUR*ones(1,5)
% ones(8,1)*[3 4 7 10 2] - NEIGHBOUR*ones(1,5)


% N1L = ones(8,1)*[i j] + [ON1 ON1].* Neighbour;
% N2L = ones(8,1)*[i j] + [ON2 ON2].* Neighbour;

%%%


%%1ST WINDOW
%------------
%k = ones(size(KeepScaledResidual,1)*size(KeepScaledResidual,2),1);
k = ones(8,length(CENTREL));
previousDiffScaledResidual1 = zeros(8,length(CENTREL),96); %nan(8,length(CENTREL),96); %By using zeros instead of NaN, in case of ON==1 we will have PARAMETER = [0;0]
FLAG_KEEP_LAST = zeros(8,length(CENTREL));
%tic
for t = 1:96
    for i = 1:length(CENTREL)
        %         sub_CENTRE= floor(CENTREL(i)/size(KeepScaledResidual,1)); %Convert from linear indexing to subscript
        %         subscript_column_CENTRE = sub_CENTRE + 1
        %         subscript_row_CENTRE = CENTREL(i) - sub_CENTRE*size(KeepScaledResidual,1)
        %CENTREL(i)
        subscript_row_CENTRE = CENTRE(i,1);
        subscript_column_CENTRE = CENTRE(i,2);
        for j = 1:8
            if ON(j,i)~=1 %Direction with no information or partial information ON(j,i) = 1
                %scaledResidual = KeepScaledResidual(:,:,timestamp-k(j,i));
                
                %The following formula is limited to number of ROW less than 1/0.00000000000000005554 in numerical software such as OCTAVE
                sub_N1= ceil(N1L(j,i)/size(KeepScaledResidual,1))-1; %Convert from linear indexing to subscript
                subscript_column_N1 = sub_N1 + 1;
                subscript_row_N1 = N1L(j,i) - sub_N1*size(KeepScaledResidual,1);
                
                %A = 612; COLUMN = ceil(A/51); ROW = A-(ceil(A/51)-1)*51;[ROW COLUMN]
                
                
                %SAMPLE = timestamp-k(j,i);
                
                %Case DIFFERENCE is NaN or Inf but I think I don't have this case, another case will be when timestamp<=96 but the file getThresholdsDiffScaledResidual is called in file contextual2.m only when timestamp>96
                SAMPLE = timestamp-k(j,i);
                % ##                if SAMPLE<1 | SAMPLE>(timestamp-1) %Case the coordinate is 0 or less, or greater than timestamp-1, then previous sample(s) is(are) choosen by reflectivity (bounce back)
                % ##                    S1 = ceil(1/2*(k(j,i) + timestamp - 1)/(timestamp - 1));
                % ##                    S2 = ceil(k(j,i)/(timestamp - 1));
                % ##                    SAMPLE = timestamp - 2*(mod(S2,2)-0.5)*(k(j,i) - 2*(S1-1)*(timestamp - 1));
                % ##                end
                
                if SAMPLE >=1
                    
                    
                    %timestamp
                    %k(j,i)
                    
                    %DIFFERENCE = scaledResidual(CENTREL(i)) - scaledResidual(N1L(j,i));
                    DIFFERENCE = KeepScaledResidual(subscript_row_CENTRE,subscript_column_CENTRE,SAMPLE) - KeepScaledResidual(subscript_row_N1,subscript_column_N1,SAMPLE);
                    
                    %Case DIFFERENCE is NaN or Inf but I think I don't have this case
                    while isnan(DIFFERENCE)==1 | isinf(DIFFERENCE)==1 %condition on k(j) so that it does not exceed the limit, also insert (active) fire detection results
                        %scaledResidual = KeepScaledResidual(:,:,timestamp-k(j,i));
                        
                        %DIFFERENCE = scaledResidual(CENTREL(i)) - scaledResidual(N1L(j,i));
                        %DIFFERENCE = KeepScaledResidual(subscript_row_CENTRE,subscript_column_CENTRE,SAMPLE) - KeepScaledResidual(subscript_row_N1,subscript_column_N1,SAMPLE);
                        
                        k(j,i) = k(j,i) + 1;
                        SAMPLE = timestamp-k(j,i);
                        if SAMPLE<1 %| SAMPLE>(timestamp-1)
                            % ##                        S1 = ceil(1/2*(k(j,i) + timestamp - 1)/(timestamp - 1));
                            % ##                        S2 = ceil(k(j,i)/(timestamp - 1));
                            % ##                        SAMPLE = timestamp - 2*(mod(S2,2)-0.5)*(k(j,i) - 2*(S1-1)*(timestamp - 1));
                            break;
                        end
                        DIFFERENCE = KeepScaledResidual(subscript_row_CENTRE,subscript_column_CENTRE,SAMPLE) - KeepScaledResidual(subscript_row_N1,subscript_column_N1,SAMPLE);
                    end
                end
                if SAMPLE < 1
                    % ##                  THRESHOLD_ACTIVE1 = LATEST_THRESHOLD_ACTIVE1;
                    FLAG_KEEP_LAST(j,i) = 1;
                    %Number of samples to define distribution is less than 96
                else
                    previousDiffScaledResidual1(j,i,t) = DIFFERENCE;
                    k(j,i) = k(j,i) + 1;
                    %                 SAMPLE = timestamp-k(j,i);
                    %                 if SAMPLE<1 | SAMPLE>(timestamp-1) %Case the coordinate is 0 or less or greater than timestamp-1, then previous sample(s) is(are) choosen by reflectivity (bounce back)
                    %                     S1 = ceil(1/2*(k(j,i) + timestamp - 1)/(timestamp - 1));
                    %                     S2 = ceil(k(j,i)/(timestamp - 1));
                    %                     SAMPLE = timestamp - 2*(mod(S2,2)-0.5)*(k(j,i) - 2*(S1-1)*(timestamp - 1));
                    %                 end
                end
            end
        end
    end
end




load logisticCoefficients;

PARAMETERS1 = zeros(8,length(CENTREL),2);
for i = 1:length(CENTREL)
    for j = 1:8
        if FLAG_KEEP_LAST(j,i)~=1 %If 96 samples are available
            pDSR = squeeze(previousDiffScaledResidual1(j,i,:));
            PARAMETERS1(j,i,1:2) = logisticCoefficients*sort(pDSR(:));
            %PARAMETERS1(:,i,:) = (logisticCoefficients*squeeze(previousDiffScaledResidual1(:,i,:)).').';
            %PARAMETERS1 = logisticCoefficients*previousDiffScaledResidual1;
            KEEP_PARAMETERS = PARAMETERS1(j,i,:);
            KEEP_PARAMETERS = KEEP_PARAMETERS(:);
            if KEEP_PARAMETERS(2)<0 %If scale parameter is negative
                FLAG_KEEP_LAST(j,i) = 1;
                PARAMETERS1(j,i,1:2) = [0;0];
            end
        end
    end
end

%THRESHOLD_COEFFICIENT_CONTEXTUAL(3): 2 out of 3 (Majority voting rule)
%THRESHOLD_COEFFICIENT_CONTEXTUAL(2): 3 out of 5 (Majority voting rule)
%THRESHOLD_COEFFICIENT_CONTEXTUAL(1): 4 out of 8

THRESHOLD_COEFFICIENT_CONTEXTUAL_MATRIX = ones(size(ON))* THRESHOLD_COEFFICIENT_CONTEXTUAL(1);
DIRECTION_WITH_INFO = 8*ones(1,length(CENTREL))- sum(ON,1);
THRESHOLD_COEFFICIENT_CONTEXTUAL_MATRIX(:,find(DIRECTION_WITH_INFO==3)) = THRESHOLD_COEFFICIENT_CONTEXTUAL(3);
THRESHOLD_COEFFICIENT_CONTEXTUAL_MATRIX(:,find(DIRECTION_WITH_INFO==5)) = THRESHOLD_COEFFICIENT_CONTEXTUAL(2);

THRESHOLD_ACTIVE1 = PARAMETERS1(:,:,1) + THRESHOLD_COEFFICIENT_CONTEXTUAL_MATRIX.*PARAMETERS1(:,:,2);
%%%

clear previousDiffScaledResidual1 PARAMETERS1

THRESHOLD_ACTIVE1 = THRESHOLD_ACTIVE1 + FLAG_KEEP_LAST.*LATEST_THRESHOLD_ACTIVE1;

%2ND WINDOW
%-----------
%%
%k = ones(size(KeepScaledResidual,1)*size(KeepScaledResidual,2),1);
k = ones(8,length(CENTREL));
previousDiffScaledResidual2 = zeros(8,length(CENTREL),96);
FLAG_KEEP_LAST = zeros(8,length(CENTREL));
%tic
for t = 1:96
    for i = 1:length(CENTREL)
        %         sub_CENTRE= floor(CENTREL(i)/size(KeepScaledResidual,1)); %Convert from linear indexing to subscript. This part is repeated
        %         subscript_column_CENTRE = sub_CENTRE + 1;
        %         subscript_row_CENTRE = CENTREL(i) - sub_CENTRE*size(KeepScaledResidual,1);
        
        subscript_row_CENTRE = CENTRE(i,1);
        subscript_column_CENTRE = CENTRE(i,2);
        
        for j = 1:8
            if ON(j,i)~=1
                %scaledResidual = KeepScaledResidual(:,:,timestamp-k(j,i));
                sub_N1= ceil(N1L(j,i)/size(KeepScaledResidual,1))-1; %Convert from linear indexing to subscript. This part is repeated
                subscript_column_N1 = sub_N1 + 1;
                subscript_row_N1 = N1L(j,i) - sub_N1*size(KeepScaledResidual,1);
                
                %                 N1L(j,i)
                
                sub_N2= ceil(N2L(j,i)/size(KeepScaledResidual,1))-1; %Convert from linear indexing to subscript
                subscript_column_N2 = sub_N2 + 1;
                subscript_row_N2 = N2L(j,i) - sub_N2*size(KeepScaledResidual,1);
                
                %                 N2L(j,i)
                %                 ON(j,i)
                
                SAMPLE = timestamp-k(j,i);
                % ##                if SAMPLE<1 | SAMPLE>(timestamp-1) %Case the coordinate is 0 or less, or greater than timestamp-1, then previous sample(s) is(are) choosen by reflectivity (bounce back)
                % ##                    S1 = ceil(1/2*(k(j,i) + timestamp - 1)/(timestamp - 1));
                % ##                    S2 = ceil(k(j,i)/(timestamp - 1));
                % ##                    SAMPLE = timestamp - 2*(mod(S2,2)-0.5)*(k(j,i) - 2*(S1-1)*(timestamp - 1));
                % ##                end
                
                if SAMPLE >=1
                    
                    %DIFFERENCE = scaledResidual(CENTREL(i)) - scaledResidual(N1L(j,i));
                    DIFFERENCE = (KeepScaledResidual(subscript_row_CENTRE,subscript_column_CENTRE,SAMPLE) + KeepScaledResidual(subscript_row_N1,subscript_column_N1,SAMPLE))/2 - KeepScaledResidual(subscript_row_N2,subscript_column_N2,SAMPLE);
                    while isnan(DIFFERENCE)==1 | isinf(DIFFERENCE)==1 %condition on k(j) so that it does not exceed the limit, also insert (active) fire detection results
                        %scaledResidual = KeepScaledResidual(:,:,timestamp-k(j,i));
                        
                        %DIFFERENCE = scaledResidual(CENTREL(i)) - scaledResidual(N1L(j,i));
                        %DIFFERENCE = KeepScaledResidual(subscript_row_CENTRE,subscript_column_CENTRE,timestamp-k(j,i)) - KeepScaledResidual(subscript_row_N1,subscript_column_N1,timestamp-k(j,i));
                        %DIFFERENCE = (KeepScaledResidual(subscript_row_CENTRE,subscript_column_CENTRE,SAMPLE) + KeepScaledResidual(subscript_row_N1,subscript_column_N1,SAMPLE))/2 - KeepScaledResidual(subscript_row_N2,subscript_column_N2,SAMPLE);
                        k(j,i) = k(j,i) + 1;
                        SAMPLE = timestamp-k(j,i);
                        if SAMPLE<1 %| SAMPLE>(timestamp-1) %Case the coordinate is 0 or less or greater than timestamp-1, then previous sample(s) is(are) choosen by reflectivity (bounce back)
                            % ##                        S1 = ceil(1/2*(k(j,i) + timestamp - 1)/(timestamp - 1));
                            % ##                        S2 = ceil(k(j,i)/(timestamp - 1));
                            % ##                        SAMPLE = timestamp - 2*(mod(S2,2)-0.5)*(k(j,i) - 2*(S1-1)*(timestamp - 1));
                            break;
                        end
                        DIFFERENCE = (KeepScaledResidual(subscript_row_CENTRE,subscript_column_CENTRE,SAMPLE) + KeepScaledResidual(subscript_row_N1,subscript_column_N1,SAMPLE))/2 - KeepScaledResidual(subscript_row_N2,subscript_column_N2,SAMPLE);
                    end
                end
                if SAMPLE < 1
                    % ##                THRESHOLD_ACTIVE2 = LATEST_THRESHOLD_ACTIVE2;
                    FLAG_KEEP_LAST(j,i) = 1;
                    %Number of samples to define distribution is less than 96
                else
                    previousDiffScaledResidual2(j,i,t) = DIFFERENCE;
                    k(j,i) = k(j,i) + 1;
                    %                 SAMPLE = timestamp-k(j,i);
                    %                 if SAMPLE<1 | SAMPLE>(timestamp-1) %Case the coordinate is 0 or less or greater than timestamp-1, then previous sample(s) is(are) choosen by reflectivity (bounce back)
                    %                     S1 = ceil(1/2*(k(j,i) + timestamp - 1)/(timestamp - 1));
                    %                     S2 = ceil(k(j,i)/(timestamp - 1));
                    %                     SAMPLE = timestamp - 2*(mod(S2,2)-0.5)*(k(j,i) - 2*(S1-1)*(timestamp - 1));
                    %                 end
                end
            end
        end
    end
end



%load logisticCoefficients;

PARAMETERS2 = zeros(8,length(CENTREL),2);
for i = 1:length(CENTREL)
    for j = 1:8
        if FLAG_KEEP_LAST(j,i)~=1 %If 96 samples are available
            pDSR = squeeze(previousDiffScaledResidual2(j,i,:));
            PARAMETERS2(j,i,1:2) = logisticCoefficients*sort(pDSR(:));
            %PARAMETERS2(:,i,:) = (logisticCoefficients*squeeze(previousDiffScaledResidual2(:,i,:)).').';
            %PARAMETERS1 = logisticCoefficients*previousDiffScaledResidual1;
            KEEP_PARAMETERS = PARAMETERS2(j,i,:);
            KEEP_PARAMETERS = KEEP_PARAMETERS(:);
            if KEEP_PARAMETERS(2)<0 %If scale parameter is negative
                FLAG_KEEP_LAST(j,i) = 1;
                PARAMETERS2(j,i,1:2) = [0;0];
            end
        end
    end
end


THRESHOLD_ACTIVE2 = PARAMETERS2(:,:,1) + THRESHOLD_COEFFICIENT_CONTEXTUAL_MATRIX .* PARAMETERS2(:,:,2); %Use THRESHOLD_COEFFICIENT_CONTEXTUAL depending of ON number different from 1
%make THRESHOLD_COEFFICIENT_CONTEXTUAL a matrix with 2 out of 3, 3 out of 5 and 4 out of 8.
%%%

THRESHOLD_ACTIVE2 = THRESHOLD_ACTIVE2 + FLAG_KEEP_LAST.*LATEST_THRESHOLD_ACTIVE2;


%scaledResidual = KeepScaledResidual(:,:,timestamp);
%DATA_N1 = scaledResidual(N1L);
%R1 = ones(8,1)*scaledResidual(CENTREL) - DATA_N1;
%DATA_N2 = scaledResidual(N2L);
%R2 = (ones(8,1)*scaledResidual(CENTREL) + DATA_N1)/2 - DATA_N2;


%[DATA_N1(:,1) DATA_N2(:,1)]

%R1 = ones(8,1)*PotentialFireDetect(CENTREL) - DATA_N1;
%R2 = (ones(8,1)*PotentialFireDetect(CENTREL) + DATA_N1)/2 - DATA_N2;
%%%


%previousDiffScaledResidual_1 = readPreviousScaledResidual(KeepScaledResidual,timestamp);


% load logisticCoefficients;
% PARAMETERS1 = logisticCoefficients*previousDiffScaledResidual_1;
% THRESHOLD_ACTIVE1 = PARAMETERS1(:,:,1) + THRESHOLD_COEFFICIENT_CONTEXTUAL*PARAMETERS1(:,:,2);
%clear lL1 sL1 previousDiffScaledResidual



%previousDiffScaledResidual_2 = readPreviousScaledResidual(KeepScaledResidual,timestamp);


%PARAMETERS2 = logisticCoefficients*previousDiffScaledResidual_2;
%THRESHOLD_ACTIVE2 = PARAMETERS2(:,:,1) + THRESHOLD_COEFFICIENT_CONTEXTUAL*PARAMETERS2(:,:,2);
%clear lL2 sL2 previousDiffScaledResidual
