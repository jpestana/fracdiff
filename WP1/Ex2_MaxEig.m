% Script for computing Table 5.5 in 
% J. Pestana, Preconditioners for symmetrized Toeplitz and multilevel
% Toeplitz matrices, 2018
%
% J. Pestana, August 3, 2018


addpath(genpath('../smt'))

L = 0; R = 1; T = 1; % Problem dimensions

% Symbol of fractional diffusion matrix
f = @(th,al) exp(-1i*th).*(1+exp(1i*(th+pi))).^al;

dlist = [0 3; 1 3; 0.5 1; 1 1]; % Diffusion parameters
alist = 1:0.25:1.75;            % Fractional derivative orders
Nlist = 2.^(10:5:30);           % Problem sizes

anum = length(alist);
dnum = size(dlist,1);

maxrat = zeros(anum,dnum); % To store bound

for j = 1:anum % Loop over alpha
    for k = 1:dnum % Loop over diffusion coefficients
        
        % Get values of quantities
        alpha = alist(j);
        d1 = dlist(k,1);
        d2 = dlist(k,2);
        N = Nlist(1);
        
        M = ceil(N^alpha); % Time step
        dx = 1/(N+1);
        dt = 1/(M+1);
        nu = dx^alpha/dt;
        
        % Symbol
        s = @(th) nu - d1*f(th,alpha) - d2*f(-th,alpha);
        st = s(linspace(-pi,pi,1000));
        
        % Upper bounds
        maxrat(j,k) = max(imag(st)./real(st));
    end
end
maxrat(maxrat<1e-14)=0;

fid = fopen('Ex2_MaxEig.tex','w+');

fprintf(fid,'$\\alpha$ &  \\multicolumn{%i}{c||}{$(d_+,d_-)$}\\\\\n',dnum);
fprintf(fid,'\\hline\n');

for k = 1:dnum
    fprintf(fid,'& (%g,%g) ',dlist(k,:));
end

fprintf(fid,'\\\\\n');

fprintf(fid,'\\hline\n');
for j = 1:anum
    fprintf(fid,'%g & %3.2f & %3.2f & %3.2f & %3.2f\\\\\n',alist(j),(maxrat(j,:)));
end
fclose(fid);
