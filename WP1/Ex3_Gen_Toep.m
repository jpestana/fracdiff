function [Axsmt,Aysmt,u0,f,hx,hy,dt] = Ex3_Gen_Toep(L,R,T,N,M,mtstep,alph,beta,dv,ev)

% [Axsmt,Aysmt,u0,f,hx,hy,dt] = Ex3_Gen_Toep(L,R,T,N,M,mtstep,alph,beta,dv,ev)
% 
% Generates the coefficient matrix A, RHS f, and initial condition for a 2D
% fractional diffusion problem with boundary condition u(x<L,t)=u(x>R,t)=0
%
% Inputs:   L,R:    Each is a 2x1 vector. [L(1), R(1)] is the x-interval of 
%                   interest, [L(2), R(2)] is the y-interval of interest
%           T:      end time
%           N:      [N(1), N(2)] are the dimensions in the x and y
%                   directions
%           mtstep: max number of time steps run
%           alph:   fractional order in x direction
%           beta:   fractional order in y direction
%           dv:     2x1 vector of diffusion coefficients in x direction
%           ev:     2x1 vector of diffusion coefficients in y direction
%
% Outputs:  Axsmt:  1D Toeplitz matrix for x direction
%           Aysmt:  1D Toeplitz matrix for y direction
%           u0:     Initial condition
%           f:      Right-hand side
%           hx:     x-direction mesh width
%           hy:     y-direction mesh width
%           dt:     time step
% 
% J. Pestana, August 3, 2018

% Step sizes and parameters
hx = (R(1)-L(1))/(N(1)+1);
hy = (R(2)-L(2))/(N(2)+1);
dt = T/M;
d1 = dv(1);
d2 = dv(2);
e1 = ev(1);
e2 = ev(2);

% Compute L_alpha
galph = cumprod([1, 1 - ((alph + 1)./(1 : N(1)))]);
ralph = dt*[(d1+d2)*galph(2),d1*galph(1) + d2*galph(3),d2*galph(4:end)]/(hx^alph);
calph = dt*[(d1+d2)*galph(2),d2*galph(1) + d1*galph(3),d1*galph(4:end)]/(hx^alph);
Axsmt = smtoep(calph,ralph);

% Compute L_beta
gbeta = cumprod([1, 1 - ((beta + 1)./(1 : N(2)))]);
rbeta = dt*[(e1+e2)*gbeta(2),e1*gbeta(1) + e2*gbeta(3),e2*gbeta(4:end)]/(hy^beta);
cbeta = dt*[(e1+e2)*gbeta(2),e2*gbeta(1) + e1*gbeta(3),e1*gbeta(4:end)]/(hy^beta);
Aysmt = smtoep(cbeta,rbeta);

% x and y co-ordinates of nodes
xx = (L(1)+hx:hx:R(1)-hx)';
yy = (L(2)+hy:hy:R(2)-hy)';
xp = kron(ones(N(2),1),xx);
yp = kron(yy,ones(N(1),1));

% Initial condition
u0 = zeros(size(xp));

% Right-hand side
f = zeros(N(1)*N(2),mtstep);
for j = 1:mtstep
    t = j*dt;
   f(:,j) = 100*sin(10*xp).*cos(yp) + sin(10*t)*xp.*yp; 
end

f = dt*f;