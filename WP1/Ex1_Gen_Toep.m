function [c,r] = Ex1_Gen_Toep(n)

% function [c,r] = Ex1_Gen_Toep(n)
%
% Generates Toeplitz matrix in Example 1 from J. Pestana, Preconditioners 
% for symmetrized Toeplitz and multilevel Toeplitz matrices, 2018.
%
% Outputs:  c: first column of Toeplitz matrix 
%           r: first row of Toeplitz matrix 
%
% J. Pestana, July 2, 2018

% Initialise c,r
c = zeros(n,1);
r = zeros(n,1);

% Loop over rows
    for k = 1:n
        t = k-1;
        s = 1-k;
        if k == 1 % First row
            r(1) = integral(@(x)def_func(x,s),-pi,pi,'RelTol',1e-12);
            c(1) = r(1);
        else % Other rows
            r(k) = integral(@(x)def_func(x,s),-pi,pi,'RelTol',1e-12);
            c(k) = integral(@(x)def_func(x,t),-pi,pi,'RelTol',1e-12);
            
        end
    end
    % Scale and remove (very small) imaginary part
c = real(c/(2*pi));
r = real(r/(2*pi));
end


function f = def_func(x,s)
% Generating function
    f = (2-2*cos(x)).*(1+1i*x).*exp(-1i*s*x);
end
