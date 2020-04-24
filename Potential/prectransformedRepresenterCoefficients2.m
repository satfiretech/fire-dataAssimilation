function iv = transformedRepresenterCoefficients(F,Ht,Q, Bo,N, wt,Rb)

%Stabilized representer matrix x representer coefficients vector


%wb = wt.';
wb = reshape(wt,size(wt,1)*size(wt,2),1);

%ab instead wb


b = backwardSweep2(F,Ht,wt,N);
g = forwardSweep2(F,Q,Bo,b,N);

for i = 1:N+1
    %p(:,i) = Ht(:,i).'*g(:,i);
    p(:,i) = Ht(:,1+(i-1)*(size(Ht,2)/(N+1)):i*(size(Ht,2)/(N+1))).'*g(:,i);
    %p(:,i) = sqrt(inv(R(:,i)))*Ht(:,i).'*g(:,i);
end
%Make a block diagonal of HPH
%pb = p.';
pb = reshape(p,size(p,1)*size(p,2),1);
%pb + sqrt(Rb)*wb = sqrt(inv(Rb))*(yb-Hb*xsfvector);
%iv = pb + wb; 
iv = pb + Rb*wb; 
%block pb
%pb + sqrt(Rb)*ab 

