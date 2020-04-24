function [xsolution] = conjugategradient(F,Ht,Q, Bo,N, R, Rb, x,b)

%flexible preconditioned conjugate gradient (Notay, 2000; Chua et al., 2009)


%14 January 2005
%a-input is the initial point
%Conjugate gradient algorithm
%[a] = conjugategradient(a,X,y,W)
% Xa = y , W is the weight vector
%The function returns a(k+1) given a(k)


%  Q = size(X,1);
%  M = size(X,2);
%  CLength = Q-M+1;
%
%  if(Q<M)
%     display('impossible: Number of sample output must be greater than number of sample input')
%  else
%     L = 2^(ceil(log2(Q)));
% end
%
% Q1 = L - Q;
% Q2 = L - M;


% X = [1 0 0  0 0 0 3 2
%      2 1 0  0 0 0 0 3
%      3 2 1  0 0 0 0 0
%      0 3 2  1 0 0 0 0
%      0 0 3  2 1 0 0 0
%
%      0 0 0  3 2 1 0 0
%      0 0 0  0 3 2 1 0
%      0 0 0  0 0 3 2 1];


% X_1 = zeros(Q,Q2);
%
% for i = 1:(CLength-1)
%     X_1(i,:) = [zeros(1,Q2-CLength+i) X(CLength:-1:(i+1),1)'];
% end
%
%
% for i= 1:(CLength-1)
%     X_1(M+i:Q,i) = X(1:(CLength-i),1);
% end
%
% X_2 = zeros(Q1,M);
% X_3 = zeros(Q1,Q2);
% for i=1:Q1
%     X_3(i,:) =[zeros(1,i-1) X(CLength:-1:1,1)' zeros(1,Q2-CLength-i+1)];
% end
%Another option is to form the big matrix Xcir immediately

% for k=1:L
%      for j=1:L
%          ADFT(k,j) = exp(-sqrt(-1)*2*pi/L*(k-1)*(j-1));
%      end
% end


% ak = [a;zeros(Q2,1)];
%
% %T1 = [Wv' zeros(1,Q1)]'.*(ADFT'*((ADFT*[X(:,1);X_2(:,1)]).*(ADFT*ak))/L);
% T1 = [Wv' zeros(1,Q1)]'.*ifft(fft([X(:,1);X_2(:,1)]).*fft(ak));
%
% %T2 = (ADFT*((ADFT*[X(:,1);X_2(:,1)]).*(ADFT'*T1))/L);
% T2 = fft(fft([X(:,1);X_2(:,1)]).*ifft(T1));
%
% T3 = T2(1:M);
%
% T4 = [Wv' zeros(1,Q1)]'.*[y; zeros(Q1,1)];
%
% %T5 = (ADFT*((ADFT*[X(:,1);X_2(:,1)]).*(ADFT'*T4))/L);
% T5 = fft(fft([X(:,1);X_2(:,1)]).*ifft(T4));
%
% T6 = [T5(1:M)];



% A = X'*W*X;
% b = X'*W*y;

%xbo = x.'; %Initialization
xbo = reshape(x,size(x,1)*size(x,2),1);
theta = 0.001;
imax = size(xbo,1)*2;

beta = zeros(imax,1);
alpha = zeros(imax,1);

xb = zeros(size(xbo,1),imax); %xb = x block
r = zeros(size(b,1),imax);  %size(b,1) = size(xbo,1) as conjugate is applied to a system with square matrix A
z = zeros(size(r));
p = zeros(size(r));
Ap = zeros(size(r));



%i=0;


i=0;
%r = b-[A*x];
ro = b - prectransformedRepresenterCoefficients2(F,Ht,Q, Bo,N,x,Rb);
%zo = sqrt(inv(Rb))*r;

%Preconditioning
%z = inv(M)*r;
%p = z
%i=0;
%p(:,1) = z(:,0);

%if (norm(p)~=0)
% if (norm(r(:,0))/norm(b)>=theta)
xsolution = xbo;
wsc = norm(ro)/norm(b);

%while norm(r(:,i))/norm(b) > theta  on ro
while wsc > theta
    %     for i = 1:imax-1,
    %         rold = r;
    %         zold = z;

    %         pk = [p;zeros(Q2,1)];
    %
    %         %T7 = [Wv' zeros(1,Q1)]'.*(ADFT'*((ADFT*[X(:,1);X_2(:,1)]).*(ADFT*pk))/L);
    %         T7 = [Wv' zeros(1,Q1)]'.*ifft(fft([X(:,1);X_2(:,1)]).*fft(pk));
    %
    %         %T8 = (ADFT*((ADFT*[X(:,1);X_2(:,1)]).*(ADFT'*T7))/L);
    %         T8 = fft(fft([X(:,1);X_2(:,1)]).*ifft(T7));
    %
    %         Ap = T8(1:M);




    %Preconditioning
    %z = inv(M)*r;
    if i==0
        zo = sqrt(inv(Rb))*ro;
        %zo = (inv(Rb))*ro;
        %zo = sqrt(inv(Rb))*r;
    else
        z(:,i) = sqrt(inv(Rb))*r(:,i);
        %z(:,i) = (inv(Rb))*r(:,i);
    end

    i = i + 1;



    if i==imax %if i==(imax-1)
        %warning('GCP:MAXIT', 'maximum iteration reached, no conversion.')
        warning('GCP:MAXIT', 'maximum iteration reached.')
        break;
    end


    if i==1
        p(:,i) = zo; %z(:,i-1);
        %beta(:,1) = z(:,0);
        %Ap(:,i) = prectransformedRepresenterCoefficients(F,Ht,Q, Bo,N, p(:,i).',Rb);
        Ap(:,i) = prectransformedRepresenterCoefficients2(F,Ht,Q, Bo,N, reshape(p(:,i),size(p,1)/(N+1),N+1),Rb);

        

        beta(i) = (zo'*Ap(:,i))/(p(:,i)'*Ap(:,i));
    else

        %beta = (z'*(r-rold))./(zold'*rold); %Polak–Ribière formula

        %beta = (r'*r)/(rold'*rold); %Fletcher–Reeves formula
        %beta = (r'*(r-rold))./(rold'*rold); %Polak–Ribière formula

        betap = zeros(size(p(:,1)));
        for l=1:i-1
            betap = betap + beta(l)*p(:,l);
        end
        p(:,i) = z(:,i-1) - betap;
        %p= r+beta*p;
        %i = i+1;

        %Ap(:,i) = prectransformedRepresenterCoefficients(F,Ht,Q, Bo,N, p(:,i).',Rb);
        Ap(:,i) = prectransformedRepresenterCoefficients2(F,Ht,Q, Bo,N, reshape(p(:,i),size(p,1)/(N+1),N+1),Rb);

        beta(i) = (z(:,i-1)'*Ap(:,i))/(p(:,i)'*Ap(:,i)); %Not Fletcher–Reeves formula
    end



    %Ap(:,i) = prectransformedRepresenterCoefficients(F,Ht,Q, Bo,N, p(:,i).',Rb);


    if i==1
        alpha(i) = (p(:,i)'*ro)/(p(:,i)'*Ap(:,i)); %1
        xb(:,i) = xbo + alpha(i)*p(:,i);

        r(:,i) = ro - alpha(i)*Ap(:,i);

    else
        alpha(i) = (p(:,i)'*r(:,i-1))/(p(:,i)'*Ap(:,i)); %1
        xb(:,i) = xb(:,i-1)+alpha(i)*p(:,i);

        r(:,i) = r(:,i-1)-alpha(i)*Ap(:,i);

    end

    %if norm(r)<theta
    %if norm(r,inf)<theta
    %         if norm(r(:,i))/norm(b)<theta
    %             break;
    %         end

    wsc = norm(r(:,i))/norm(b);

    xsolution = xb(:,i);
end
% end

% if i==(imax-1)
%     %warning('GCP:MAXIT', 'maximum iteration reached, no conversion.')
%     warning('GCP:MAXIT', 'maximum iteration reached.')
% end

%i+1
%xb(:,i) %xb for x block
% xsolution %xb for x block


%Use of the Fletcher-Reeves formula for \beta
%Polak-Ribi{\`e}re formula


% [eye(M) zeros(M,Q2)]*[X X_1;X_2 X_3]'*[W zeros(Q,Q1);zeros(Q1,Q) zeros(Q1,Q1)]*[X X_1;X_2 X_3]*[a;zeros(Q2,1)]=...
%     [eye(M) zeros(M,Q2)]*[X X_1;X_2 X_3]'*[W zeros(Q,Q1);zeros(Q1,Q) zeros(Q1,Q1)]*[y;zeros(Q1,1)]
%
