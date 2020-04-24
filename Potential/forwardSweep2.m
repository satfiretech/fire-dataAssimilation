function g = forwardSweep(F,Q,Bo,b,N)


%g(:,0+1) = Pb(:,0+1)*b(:,0+1);
%g(:,0+1) = Bo*b(:,0+1);
%g(:,0+1) = Bo*b(:,0+1);
g(:,0+1) = Bo*F.'*b(:,0+2);



for n = 1+1:N+1
    %g(:,n) = F(:,n-1)*g(:,n-1) + Q(:,n)*b(:,n);
    %%g(:,n) = F(:,n)*g(:,n-1) + Q(:,n-1)*b(:,n);
    %g(:,n) = F(:,(n-1)*size(F,1)+1:n*size(F,1))*g(:,n-1) + Q(:,(n-2)*size(Q,1)+1:(n-1)*size(Q,1))*b(:,n);
    g(:,n) = F*g(:,n-1)+ Q(:,(n-2)*size(Q,1)+1:(n-1)*size(Q,1))*b(:,n);
end
