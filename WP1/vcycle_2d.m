function [u] = vcycle_2d(IAxm,IAym,IIm,diagel,Lm,f,num_presteps,num_poststeps,num_cycles,num_levels,omega)

% [u] = vcycle_2d(IAxm,IAym,IIm,diagel,Lm,u,f,num_presteps,num_poststeps,num_cycles,num_levels,omega)
%
% Performs multigrid V cycle method for a multilevel Toeplitz matrix
% A = kron(I,IAx) + kron(IAy,I). Requires quantities computed in
% vcycle_2D_setup.m. Non-recursive algorithm.
%
% Inputs:   IAxm:           Representation of IAx at each level
%           IAym:           Representation of IAy at each level
%           IIm:            Representation of identity matrix at each level
%           diagel:         element on diagonal of coarse grid Toeplitz
%                           matrix at each level
%           Lm:             1D Projection operator at each level
%           f:              right-hand side
%           num_presteps:   number of damped Jacobi pre-smoothing steps
%           num_poststeps:  number of damped Jacobi post-smoothing steps
%           num_cycles:     number of V-cycles
%           num_levels:     number of levels in V-cycle
%           omeg:           damped Jacobi damping parameter
%
% Outputs:  u:              result of V-cycle
%
% J. Pestana, August 3, 2018


% Weighting for damped Jacobi
n = sqrt(length(f));
u = zeros(size(f));

% Get fine grid quantities
dA = diagel(1);
II = IIm{1};
IAx = IAxm{1};
IAy = IAym{1};

% Loop over MG V-cycles
for cycle = 1:num_cycles
    
    % Pre-smoothing by damped Jacobi
    for k = 1:num_presteps
        u = u + (omega/dA)*(f-matvecmult(IAx,IAy,II,n,u));
    end
    
    r = f - matvecmult(IAx,IAy,II,n,u);
    
    % Restriction operator
    N = (n-1)/2;
    L = Lm{1};
    
    % Coarsen r, A
    Rc = L'*reshape(r,n,n)*L;
    
    % Coarse grid solve
    if num_levels == 1
        IIc = IIm{2};
        IAxc = IAxm{2};
        IAyc = IAym{2};
        
        Ac = kron(IAyc,IIc) + kron(IIc,IAxc);
        ec = Ac\Rc(:);
        
    else
        ec = vcycle_2d(IAxm(2:num_levels+1),IAym(2:num_levels+1),IIm(2:num_levels+1),diagel(2:end),Lm(2:num_levels),Rc(:),num_presteps,num_poststeps,1,num_levels-1,omega);
    end
    
    % Prolong u
    e = L*reshape(ec,N,N)*L';
    u = u + e(:);
    
    % Post-smoothing by damped Jacobi
    for k = 1:num_poststeps
        u = u + omega*((f-matvecmult(IAx,IAy,II,n,u))/dA);
    end
end
end

% Matrix-vector multiplication
function y = matvecmult(IAx,IAy,II,n,x)
X = reshape(x,n,n);
y = reshape(IAx*X*II'  + II*X*(IAy'),n^2,1);
end

