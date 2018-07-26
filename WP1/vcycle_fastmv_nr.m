function [u] = vcycle_fastmv_nr(diagel,Asm,Ac,f,num_presteps,num_poststeps,num_levels,omeg)

% function [u] = vcycle_fastmv_nr(diagel,Asm,Ac,f,num_presteps,num_poststeps,num_levels,omeg)
%
% Performs 1 multigrid V cycle for a Toeplitz matrix. Requires quantities
% computed in vcycle_fastmv_setup.m. Non-recursive algorithm. Assumes 0
% initial guess.
%
% Inputs:   diagel:         vector of diagonal elements of Toeplitz
%                           matrices at each level
%           Asm:            each column contains the eigenvalues of the
%                           circulant embedding of the Toeplitz matrix at
%                           each level
%           Ac:             Toeplitz matrix at coarsest level
%           f:              right-hand side
%           num_presteps:   number of damped Jacobi pre-smoothing steps
%           num_poststeps:  number of damped Jacobi post-smoothing steps
%           num_levels:     number of levels in V-cycle
%           omeg:           damped Jacobi damping parameter
%
% Outputs:  u:              result of V-cycle
%
% J. Pestana, July 26, 2018


% Setup
n = length(f);

if n < 7
    error('Matrix too small to run multigrid')
end

RS = zeros(n,num_levels+1); % To store residual at each level
US = zeros(n,num_levels);   % To store u at each level

% Store residual at fine level
r = f;
RS(:,1) = r;

% Loop over levels to restrict
for j = 1:num_levels
    
    % Setup
    n = size(r,1);
    vec = Asm(1:2*n,j);
    z = zeros(2*n,1);
    u = zeros(n,1);
    
    % Pre-smoothing by damped Jacobi
    for k = 1:num_presteps
        dx = (omeg/diagel(j))*r;
        u = u + dx;
        z(1:n) = dx;
        tempvec = ifft(vec.*fft(z));
        r = r - real(tempvec(1:n));
    end
    US(1:n,j) = u; % Store u
    
    % Restriction operator
    N = (n-1)/2;
    
    % Coarsen r
    r = (r(1:2:n-2) + 2*r(2:2:n-1) + r(3:2:n))/4;
    RS(1:N,j+1) = r; % Store r
end

u = Ac\r; % Coarse grid solve

% Loop over levels in reverse order to prolongate
for j = num_levels:-1:1
    
    % Prolong u
    ur = reshape(repmat(u',2,1),[],1);
    dx = ([0;ur] + [ur;0])/2;
    n = size(dx,1);
    u = US(1:n,j) + dx;
    
    vec = Asm(1:2*n,j);
    z = zeros(2*n,1);
    z(1:n) = u;
    tempvec = ifft(vec.*fft(z));
    r = RS(1:n,j)-real(tempvec(1:n));
    
    % Post-smoothing by damped Jacobi
    for k = 1:num_poststeps
        dx = (omeg/diagel(j))*r;
        u = u + dx;
        z(1:n) = dx;
        tempvec = ifft(vec.*fft(z));
        r = r - real(tempvec(1:n));
    end
end
end
