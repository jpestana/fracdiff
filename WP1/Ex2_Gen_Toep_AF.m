function [c,r] = Ex2_Gen_Toep_AF(n,nu,alpha,d1,d2,m)

% [c,r] = Ex2_Gen_Toep_AF(n,nu,alpha,d1,d2,m)
%
% Generates the coefficient matrix A(|f|) for the 1D fractional
% diffusion problem in Example 5.1 of J. Pestana, Preconditioners for 
% symmetrized Toeplitz and multilevel Toeplitz matrices, 2018.
% 
% Inputs:   n:      problem dimension
%           nu:     tau/h^alpha, where tau is the time step and h is the 
%                   mesh width
%           alpha:  fractional diffusion parameter
%           d1,d2:  diffusion coefficients
%           m:      number of elements of r,c to compute
% 
% Outputs:  r:      first row of Toeplitz matrix
%           c:      first column of Toepltiz matrix
% 
% J. Pestana, August 3, 2018

% Initialise column and row
c = zeros(n,1);
r = zeros(n,1);

% Loop over column/row and compute Fourier coefficient
for k = 1:m
    t = k-1;
    s = 1-k;
    if k == 1
        r(1) = integral(@(x)def_func(x,nu,alpha,d1,d2,s),-pi,pi,'RelTol',1e-12);
        c(1) = r(1);
    else
        r(k) = integral(@(x)def_func(x,nu,alpha,d1,d2,s),-pi,pi,'RelTol',1e-12);
        c(k) = integral(@(x)def_func(x,nu,alpha,d1,d2,t),-pi,pi,'RelTol',1e-12);
        
    end
end
c = real(c/(2*pi));
r = real(r/(2*pi));
end

% Defines function f
function f = def_func(x,nu,alpha,d1,d2,s)
f = nu-d1*exp(-1i*x).*((1-exp(1i*x)).^alpha)-d2*exp(1i*x).*((1-exp(-1i*x)).^alpha);
f = abs(f);
f = f.*exp(-1i*s*x);
end