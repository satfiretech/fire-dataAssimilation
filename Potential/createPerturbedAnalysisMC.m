function filtered = createPerturbedAnalysis(Keepxs,NUMBER_ENSEMBLE_MEMBERS,Ao)

%Keepxs = 6 x Pixels x Timestamps
%filtered = 6 x Pixels x NUMBER_ENSEMBLE_MEMBERS
%scaled lagged average forecasting


%Ft = 1;

unperturbed = Keepxs(:,:,size(Keepxs,3));  %control
j = ones(1,size(Keepxs,2));
filtered(:,:,j(1)) = unperturbed;

for pixelnumber = 1:size(Keepxs,2)
    for i = 1:(NUMBER_ENSEMBLE_MEMBERS-1)/2
        %AA = randn(6,1);BB = AA/norm(AA);PP = chol(Ao)*BB;norm(PP) [PP is varying, use then another option]
        %randomseed = randn(6,1);
        %perturbation = chol(Ao(1:6,1+(pixelnumber-1)*6:pixelnumber*6))*randomseed/norm(randomseed);
        
        %randomseed = chol(Ao(1:6,1+(pixelnumber-1)*6:pixelnumber*6))*randn(6,1);
        %perturbation = randomseed/norm(randomseed)*norm(chol(Ao(1:6,1+(pixelnumber-1)*6:pixelnumber*6)),'fro');%/1.3;
        
        randomseed = randn(6,1);
        %perturbation = randomseed/norm(chol(Ao(1:6,1+(pixelnumber-1)*6:pixelnumber*6))*randomseed)*2;
        perturbation = randomseed/norm(chol(inv(Ao(1:6,1+(pixelnumber-1)*6:pixelnumber*6)))*randomseed);%/2;
        %perturbation = perturbation*2;%/2;
        
        %perturbation =
        %chol(Ao(1:6,1+(pixelnumber-1)*6:pixelnumber*6))*randomseed; %Perturbations of the same variance
        %http://stats.stackexchange.com/questions/120179/generating-data-wi
        %th-a-given-sample-covariance-matrix
        %perturbation = randomseed/norm(chol((Ao(1:6,1+(pixelnumber-1)*6:pixelnumber*6)))*randomseed);%/2;
        
        
        %perturbation of size
        filtered(:,pixelnumber,j(pixelnumber)+1) = unperturbed(:,pixelnumber) + perturbation;
        filtered(:,pixelnumber,j(pixelnumber)+2) = unperturbed(:,pixelnumber) - perturbation;
        j(pixelnumber) = j(pixelnumber) + 2;
    end
end

            