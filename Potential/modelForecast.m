function x_n = modelForecast(xn_1)

%F = [nan(7) eye(7) eye(7) eye(7) eye(7) eye(7)];
%Fb = (blockdiag(F,N+1)).'; [block diagonal]

F = 1;
F = eye(size(xn_1,1));
x_n = F*xn_1;


