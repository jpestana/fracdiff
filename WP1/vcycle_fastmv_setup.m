function [diagel,Asmtmult,Ac] = vcycle_fastmv_setup(col,row,num_levels)

% function [diagel,Asmtmult,Ac] = vcycle_fastmv_setup(col,row,num_levels)
% 
% Computes quantities required by vcycle_fastmv_nr for V-cycle multigrid 
% applied to a Toeplitz matrix A
%
% Inputs:   col:        first column of Toeplitz matrix A
%           row:        first row of Toeplitz matrix A
%           num_levels: number of levels in V-cycle
%
% Outputs:  diagel:     element on diagonal of coarse grid Toeplitz matrix 
%                       at each level
%           Asmtmult:   eigenvalues of circulant embedding of coarse grid 
%                       Toeplitz matrix at each level
%           Ac:         Toeplitz matrix at coarsest level
%
% J. Pestana, July 26, 2018

% Set up vectors to store quantities
n = length(row);
diagel(1:num_levels,1) = 0;
Asmtmult(2*n,num_levels) = 0;

% Fine grid level quantities
diagel(1) = col(1);
Asmtmult(:,1) = fft([col ; 0 ; row(n:-1:2)]);

% Loop over levels
for k = 1:num_levels-1   
    
    % Compute first row and column of coarse grid matrix
    ctemp = (col(3:2:n) + 4*col(2:2:n-1) + 6*col(1:2:n-2) + 4*[row(2);col(2:2:n-3)] + [row(3);col(1:2:n-4)])/8;
    row = (row(3:2:n) + 4*row(2:2:n-1) + 6*row(1:2:n-2) + 4*[col(2);row(2:2:n-3)] + [col(3);row(1:2:n-4)])/8;
    col = ctemp;
    n = length(row);
    row(1) = col(1);
    
    % Store diagonal element and row, col of coarse grid matrix
    diagel(k+1) = col(1); 
    Asmtmult(1:2*length(row),k+1) = fft([col ; 0 ; row(n:-1:2)]);
end

    % Compute Ac
    ctemp = (col(3:2:n) + 4*col(2:2:n-1) + 6*col(1:2:n-2) + 4*[row(2);col(2:2:n-3)] + [row(3);col(1:2:n-4)])/8;
    row = (row(3:2:n) + 4*row(2:2:n-1) + 6*row(1:2:n-2) + 4*[col(2);row(2:2:n-3)] + [col(3);row(1:2:n-4)])/8;
    col = ctemp;
    row(1) = col(1);
    Ac = toeplitz(col,row);
end