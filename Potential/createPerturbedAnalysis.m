function filtered = createPerturbedAnalysis(Keepxs,NUMBER_ENSEMBLE_MEMBERS,Ao)

%Keepxs = 6 x Pixels x Timestamps
%filtered = 6 x Pixels x NUMBER_ENSEMBLE_MEMBERS
%scaled lagged average forecasting

Ft = 1;

unperturbed = Keepxs(:,:,size(Keepxs,3));  %control
j = 1;
filtered(:,:,j) = unperturbed;
for i = 1:(NUMBER_ENSEMBLE_MEMBERS-1)/2
    %     filtered(:,:,j+1) = unperturbed + (Keepxs(:,:,size(Keepxs,3)-i)*Ft^i - unperturbed)/i;
    %     filtered(:,:,j+2) = unperturbed - (Keepxs(:,:,size(Keepxs,3)-i)*Ft^i - unperturbed)/i;

    %PRODUCT = 1/(1/norm(chol(inv(Ao))*(Keepxs(:,:,size(Keepxs,3)-i)*Ft^i - unperturbed))*norm(chol(inv(Ao))*(Keepxs(:,:,size(Keepxs,3)-1)*Ft^1 - unperturbed)))
    %PRODUCT = 1/(1/norm(Keepxs(:,:,size(Keepxs,3)-i)*Ft^i - unperturbed)*norm(Keepxs(:,:,size(Keepxs,3)-1)*Ft^1 - unperturbed));
    %round((PRODUCT - 1)/(i-1))
    ss = 1;
    %ss = 0.5;
    %ss = 0.3666;
    %ss = 0.3333;
    %ss = 0.25;
    %ss = 0;


    filtered(:,:,j+1) = unperturbed + (Keepxs(:,:,size(Keepxs,3)-i)*Ft^i - unperturbed)/(1 + (i-1)*ss);
    filtered(:,:,j+2) = unperturbed - (Keepxs(:,:,size(Keepxs,3)-i)*Ft^i - unperturbed)/(1 + (i-1)*ss);
    %

    %Only for a single pixel
    %     filtered(:,:,j+1) = unperturbed + (Keepxs(:,:,size(Keepxs,3)-i)*Ft^i - unperturbed)/norm(Keepxs(:,:,size(Keepxs,3)-i)*Ft^i - unperturbed)*norm(Keepxs(:,:,size(Keepxs,3)-1)*Ft^1 - unperturbed);
    %     filtered(:,:,j+2) = unperturbed - (Keepxs(:,:,size(Keepxs,3)-i)*Ft^i - unperturbed)/norm(Keepxs(:,:,size(Keepxs,3)-i)*Ft^i - unperturbed)*norm(Keepxs(:,:,size(Keepxs,3)-1)*Ft^1 - unperturbed);
%For many pixels
%     for k=1:size(Keepxs,2)
%         filtered(:,k,j+1) = unperturbed(:,k) + (Keepxs(:,k,size(Keepxs,3)-i)*Ft^i - unperturbed(:,k))/norm(Keepxs(:,k,size(Keepxs,3)-i)*Ft^i - unperturbed(:,k))*norm(Keepxs(:,k,size(Keepxs,3)-1)*Ft^1 - unperturbed(:,k));
%         filtered(:,k,j+2) = unperturbed(:,k) - (Keepxs(:,k,size(Keepxs,3)-i)*Ft^i - unperturbed(:,k))/norm(Keepxs(:,k,size(Keepxs,3)-i)*Ft^i - unperturbed(:,k))*norm(Keepxs(:,k,size(Keepxs,3)-1)*Ft^1 - unperturbed(:,k));
%     end
    
    j = j + 2;
end

