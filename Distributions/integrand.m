function y = integrand(x)


%y = x.*exp(x).*exp(-2*exp(x));
y = 2*x.*exp(x).*exp(-exp(x)).^2;
