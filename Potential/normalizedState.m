function X = normalizedState(Q,xt)

for i = 1:size(xt,2)%Number of pixels
    for j = 1:size(xt,3)%Ns: number of particles
        %Q(:,1+(i-1)*6:6+(i-1)*6)
        X(:,i,j) = chol(Q(:,1+(i-1)*6:6+(i-1)*6))*xt(:,i,j);
        %X(:,i,j) = X(:,i,j)/norm(X(:,i,j));
                
        %X(:,i,j) = chol(Q(:,1+(i-1)*6:6+(i-1)*6))'*xt(:,i,j);
       
       %X(:,i,j) = sqrtm(Q(:,1+(i-1)*6:6+(i-1)*6))*xt(:,i,j);
       %X(:,i,j) = chol(Q(:,1+(i-1)*2:2+(i-1)*2))*xt(:,i,j);
    end
end
%process_noise = extend(sqrt(Q),Ns).*randn(size(xf));  

