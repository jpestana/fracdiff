function [Asmt,u0,f,dx,nu,c,r,cl,rl] = Ex2_Gen_Toep(L,R,T,N,M,maxtimepoints,d1,d2,alpha,prob)
% [Asmt,u0,f,dx,nu,c,r,cl,rl] = Ex2_Gen_Toep(L,R,T,N,M,maxtimepoints,d1,d2,alpha,prob)
%
% Generates the coefficient matrix A, RHS f, and initial condition for a 1D
% fractional diffusion problem with boundary condition u(x<L,t)=u(x>R,t)=0
%
% [Asmt,u0,f,dx,nu,c,r,ct,rt] = genFracDiffMatricesOneD(L,R,T,N,M,mtstep,d1,d2,alpha,prob)
%
% Inputs:   L,R             Left and right endpoints of domain
%           T               End time
%           N               Number of spatial grid points
%           M               Total number of time points
%           maxtimepoints   Number of time points to compute
%           d1, d2          Left- and right-hand diffusion coefficients
%           alpha           Fractional diffusion parameter
%           prob            Example number
%
% Outputs:  Asmt            Coefficient matrix in smt format
%           u0              Initial condition
%           f               RHS
%           dx              Spatial step size
%           nu              Discretisation parameter dt/(dx^\alpha)
%           c               column values of Asmt
%           r               row values of Asmt
%           cl              column values of fractional diffusion matrix
%           rl              row values of fractional diffusion matrix
%
% J. Pestana 01/06/2018

%%%%% Coefficient matrix %%%%%

% Step sizes and parameters
dx = (R-L)/(N+1);
dt = T/(M+1);
nu = dx^alpha/dt;

% g values (for matrix L)
g = cumprod([1, 1 - ((alpha + 1)./(1 : N))]);

% Row and column of Toeplitz matrix L
rl = [g(2),g(1),zeros(1,N-2)];
cl = g(2:N+1);

% Row and column of Toeplitz coefficient matrix
r = [nu zeros(1,N-1)]-[(d1+d2)*g(2),d1*g(1) + d2*g(3),d2*g(4:N+1)];
c = [nu zeros(1,N-1)]-[(d1+d2)*g(2),d1*g(3) + d2*g(1),d1*g(4:N+1)];

% Toeplitz coefficient matrix
Asmt = smtoep(c,r);


%%%%% RHS and initial condition %%%%%
xp = (L+dx:dx:R-dx)';

if prob == 1
    % Pang and Sun
    xc = 1.2;
    sigma = 0.08;
    u0 = exp(-(xp-xc).^2/(2*sigma^2));
    f = sparse(N,maxtimepoints);
elseif prob == 2
    % Lei and Sun
    xc = 1.2;
    sigma = 0.08;
    u0 = exp(-(xp-xc).^2/(2*sigma^2));
    f = sparse(N,maxtimepoints);
elseif prob == 3
    % BSS
    u0 = zeros(size(xp));
    f = 80*sin(20*xp).*cos(10*xp);
    f = repmat(f,1,maxtimepoints);
elseif prob == 4
    % BSS+PS
    xc = 1.2;
    sigma = 0.08;
    u0 = zeros(size(xp));
    f = exp(-(xp-xc).^2/(2*sigma^2));
    f = repmat(f,1,maxtimepoints);
end