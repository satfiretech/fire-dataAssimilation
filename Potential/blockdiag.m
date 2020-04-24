function Xb = blockdiag(X,n)

Xb = zeros(size(X,1)*n,size(X,2));

for i=1:n
    %Xb(1:size(X,1),1:size(X,2)/n) = X(1:size(X,1),1:size(X,2)/n);
    Xb((i-1)*size(X,1)+1:i*size(X,1),(i-1)*(size(X,2)/n)+1:i*(size(X,2)/n)) = X(1:size(X,1),(i-1)*(size(X,2)/n)+1:i*(size(X,2)/n));
end
    