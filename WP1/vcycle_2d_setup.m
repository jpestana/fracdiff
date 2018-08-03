function [diagel,Lm,IIm,IAxm,IAymt] = vcycle_2d_setup(IAx,IAy,num_levels)

% function [diagel,Lm,IIm,IAxm,IAymt] = vcycle_2d_setup(IAx,IAy,num_levels)
%
% Computes quantities required by vcycle_2D for V-cycle multigrid
% applied to a 2D Toeplitz matrix A = kron(I,IAx) + kron(IAy,I)
%
% Inputs:   IAx:        x-dimensional part of A
%       :   IAy:        y-dimensional part of A
%       :   num_levels: number of levels in V-cycle
%
% Outputs:  diagel:     element on diagonal of coarse grid Toeplitz matrix
%                       at each level
%           Lm:         1D Projection operator at each level
%           IIm:        Representation of identity matrix at each level
%           IAxm:       Representation of IAx at each level
%           IAym:       Representation of IAy at each level


% Setup
n = size(IAx,1);

% Set up objects to store quantities
diagel(1:num_levels,1) = 0;
Lm = cell(num_levels,1);
IIm = cell(num_levels+1,1);
IAxm = cell(num_levels+1,1);
IAymt = cell(num_levels+1,1);

% Store fine grid quantities
II = speye(n);
IIm{1} = II;
IAxm{1} = IAx;
IAymt{1} = IAy;

% Extract first row and column of fine grid matrices
colI = II(:,1);
rowI = II(1,:)';
colx = IAx(:,1);
rowx = IAx(1,:)';
coly = IAy(:,1);
rowy = IAy(1,:)';

% Loop over levels
for k = 1:num_levels
    N = (n-1)/2;

    % Diagonal elements
    diagel(k) = colI(1)*(colx(1,1) + coly(1,1));


    % Compute 1D projection L at each level
    I = eye(N);
    K = zeros(n,N);
    K(2:2:end-1,:) = I;
    L = gallery('tridiag',n,0.5,1,0.5)*K;
    Lm{k} = L;
    
    % Coarsen A
    rI = (rowI(3:2:n) + 4*rowI(2:2:n-1) + 6*rowI(1:2:n-2) + 4*[colI(2);rowI(2:2:n-3)] + [colI(3);rowI(1:2:n-4)])/4;
    colI = (colI(3:2:n) + 4*colI(2:2:n-1) + 6*colI(1:2:n-2) + 4*[rowI(2);colI(2:2:n-3)] + [rowI(3);colI(1:2:n-4)])/4;
    rowI = rI; 
    IIm{k+1} = sptoeplitz(colI,rowI);
    
    rx = (rowx(3:2:n) + 4*rowx(2:2:n-1) + 6*rowx(1:2:n-2) + 4*[colx(2);rowx(2:2:n-3)] + [colx(3);rowx(1:2:n-4)])/4;
    colx = (colx(3:2:n) + 4*colx(2:2:n-1) + 6*colx(1:2:n-2) + 4*[rowx(2);colx(2:2:n-3)] + [rowx(3);colx(1:2:n-4)])/4;
    rowx = rx; rowx(1) = colx(1);
    IAxm{k+1} = toeplitz(colx,rowx);
    
    ry = (rowy(3:2:n) + 4*rowy(2:2:n-1) + 6*rowy(1:2:n-2) + 4*[coly(2);rowy(2:2:n-3)] + [coly(3);rowy(1:2:n-4)])/4;
    coly = (coly(3:2:n) + 4*coly(2:2:n-1) + 6*coly(1:2:n-2) + 4*[rowy(2);coly(2:2:n-3)] + [rowy(3);coly(1:2:n-4)])/4;
    rowy = ry; rowy(1) = coly(1);
    IAymt{k+1} = toeplitz(coly,rowy); 
    
    n = N;
end