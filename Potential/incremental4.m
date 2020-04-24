function increment = incremental(Gf,alpha, alphac)

increment = zeros(size(Gf,1),size(alpha,1)*size(alpha,2),size(alpha,3));
Ne = size(alpha,3);

%alphaAll = [reshape(alpha,1,size(alpha,1)*size(alpha,2),Ne)
%reshape(alphac,1,size(alphac,1)*size(alphac,2),Ne)];
alphaAll = zeros(2,size(alpha,1)*size(alpha,2),Ne);
alphaAll(1,:,:) = reshape(alpha,1,size(alpha,1)*size(alpha,2),Ne);
alphaAll(2,:,:) = reshape(alphac,1,size(alphac,1)*size(alphac,2),Ne);


% for i=1:size(Gf,1)
%     increment(i,:,:) = extend(Gf(i,:),Ne).*alpha;
% end


for i=1:size(alpha,1)*size(alpha,2)
    %increment(i,:,:) = extend(Gf(i,:),Ne).*alpha;
    for j = 1:Ne
        increment(1:6,i,j) = Gf(1:6,1+(i-1)*2:2+(i-1)*2)*alphaAll(:,i,j);
    end
end

