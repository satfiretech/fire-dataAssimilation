function [xa g] = postMultiplier(F,Ht,wt,Q1,Bo,N,xsf)

%Step 4: 
%Having w
%if j==1
    Q = [Q1 Q1 Q1 Q1 Q1 Q1];% Q1 Q1];%*0+1;
%else
%    Q = [Q1 Q1 Q1 Q1 Q1 Q1]*0;
%end

b = backwardSweep2(F,Ht,wt,N); 
g = forwardSweep2(F,Q,Bo,b,N);

%Step 5:
xa = xsf + g; 

%gnorm = norm(g)