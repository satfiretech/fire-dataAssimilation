function b = backwardSweep(F,Ht,wt,N)

%b(:,N+1) = H(:,N+1)'*w(:,N+1);

b = zeros(size(Ht,1),N+1);
%b = zeros(size(F,1),N);

%b(:,N+1) = Ht(:,N+1)*wt(:,N+1);  %0 for Uboldi
b(:,N+1) = Ht(:,1+((N+1)-1)*2:(N+1)*2)*wt(:,N+1);  %0 for Uboldi





for n = N:-1:1
    %size(F(:,n))
    %size(Ht(:,n))
    %size(wt(:,n))
    %b(:,n) = F(:,(n)*size(F,1)+1:(n+1)*size(F,1))'*b(:,n+1) + Ht(:,n)*wt(:,n);
    %b(:,n) = F'*b(:,n+1) + Ht(:,n)*wt(:,n);
    b(:,n) = F'*b(:,n+1) + Ht(:,1+(n-1)*2:n*2)*wt(:,n);
    
end
%bo = b(:,1);



