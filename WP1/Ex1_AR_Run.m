function [iv,tv,rv] = Ex1_AR_Run

% function [it,tv,rv] = Ex1_AR_Run
%
% Example 1, Table 5.1 from J. Pestana, Preconditioners for symmetrized 
% Toeplitz and multilevel Toeplitz matrices, 2018.
%
% Outputs:  iv: array of iteration numbers
%           tv: array of CPU times
%           rv: array of relative residuals
%
% J. Pestana, July 26, 2018

%%% Setup %%%
addpath('../smt') % Add circulant preconditioner directory to path
addpath('./Ex1Data') % Add Toeplitz coefficient files to path

fid = fopen('Ex1_AR.txt','w+'); % Open file for table
eigcalc = 0;  % 1 for eigenvalue computation, 0 otherwise
circtype = 'optimal'; % Use optimal circulant preconditioner

nvec = 2.^(9:13)-1;  % Dimensions of problems

% Set up arrays to store iterations, etc.
n_time = length(nvec);
iv = zeros(n_time,6);
tv = zeros(n_time,6);
rv = zeros(n_time,6);

rng('default'); % Set random number generator

%%% Main computations
for j = 1:n_time
    n = nvec(j);
    disp(n)
    fprintf('Building matrix...')
    load(['Ex1_cr_',num2str(n),'.mat'],'c','r');
    
    r(1) = c(1);                    % Consistency requirement
    Tsmt = smtoep(c,r);
    Tsmtmult = toeprem(Tsmt);
    fprintf('Done\n');
    
    b = randn(n,1); b = b/norm(b);  % RHS             
    tol = 1e-8;                     % Iterative solver tolerance
    x0 = ones(n,1)/sqrt(n);         % Intial guess
    
    % Set up preconditioners
    
    
    % GMRES with Strang circulant
    fprintf('GMRES with circulant...')
    tic;
    Csmt = smtcprec(circtype,Tsmt);
    [~, ~, itgpc, ~,resgpc] = rpgmres( @(x)tmtimes(Tsmtmult,x,'notransp'), b, tol, n, 1, @(y)capply(Csmt,y), x0 );
    tv(j,1) = toc;
    rv(j,1) = resgpc(end);
    iv(j,1) = itgpc(2);
    fprintf('Done\n');
    
    % GMRES with tridiagonal preconditioner
    fprintf('GMRES with tridiagonal preconditioner...')
    tic;
    P = gallery('tridiag',n,-1,2,-1);
    [~, ~, itgp, ~,resgp] = rpgmres( @(x)tmtimes(Tsmtmult,x,'notransp'), b, tol, n, 1,P, x0 );
    tv(j,2) = toc;
    rv(j,2) = resgp(end);
    iv(j,2) = itgp(2);
    fprintf('Done\n');
    
    % LSQR with circulant preconditioner
    fprintf('LSQR with circulant preconditioner...')
    tic;
    Csmt = smtcprec(circtype,Tsmt);
    Csmttrans = smcirc(Csmt(1,:)');
    [~, ~, ~,iv(j,3), reslpc] = lsqr(@(x,transp_flag)tmtimes(Tsmtmult,x,transp_flag),b,tol,n,@(y,transp_flag)capplytrans(Csmt,Csmttrans,y,transp_flag),[],x0);
    tv(j,3) = toc;
    rv(j,3) = reslpc(end);
    fprintf('Done\n');
    
    % LSQR with tridiagonal preconditioner
    fprintf('LSQR with tridiagonal preconditioner...')
    tic;
    P = gallery('tridiag',n,-1,2,-1);
    [~, ~, ~,iv(j,4), reslp] = lsqr(@(x,transp_flag)tmtimes(Tsmtmult,x,transp_flag),b,tol,n,P,[],x0);
    tv(j,4) = toc;
    rv(j,4) = reslp(end);
    fprintf('Done\n');
    
    % MINRES with circulant preconditioner
    fprintf('MINRES with circulant preconditioner...')
    tic;
    Csmt = smtcprec(circtype,Tsmt);
    CPsmt =  smcirc(ifft(abs(eig(Csmt))));
    Yb = b(n:-1:1);                 % Apply Y to b
    [~,~,~,iv(j,5),resmpc] = minres(@(x)ymtimes(Tsmtmult,x),Yb,tol,n,@(y)capply(CPsmt,y),[],x0);
    tv(j,5) = toc;
    rv(j,5) = resmpc(end);
    fprintf('Done\n');
    
    % MINRES with tridiagonal preconditioner
    fprintf('MINRES with tridiagonal preconditioner...')
    tic;
    P = gallery('tridiag',n,-1,2,-1);
    Yb = b(n:-1:1);                 % Apply Y to b
    [~,~,~,iv(j,6),resmp] = minres(@(x)ymtimes(Tsmtmult,x),Yb,tol,n,P,[],x0);
    tv(j,6) = toc;
    rv(j,6) = resmp(end);
    fprintf('Done\n');
    save Ex1_AR iv tv rv
   
    % Add iterations and timings to table
    fprintf(fid,'%i & %i & (%3.2g) & %i & (%3.2g) & %i & (%3.2g) & %i & (%3.2g) & %i & (%3.2g) & %i & (%3.2g) \\\\\n',n,iv(j,1),tv(j,1),iv(j,2),tv(j,2),iv(j,3),tv(j,3),iv(j,4),tv(j,4),iv(j,5),tv(j,5),iv(j,6),tv(j,6));
    
    % Eigenvalue computations
    if n == 2047 && eigcalc == 1
        eac = eig(full(Tsmt),full(P));
        eyac = eig(flipud(full(Tsmt)),full(P));
        
        figure('visible','off');
        hold on;
        plot(real(eac),imag(eac),'x');
        hold off;
        set(gca,'fontsize',20);
        xlim([0 2])
        set(gca,'TickLabelInterpreter','latex')
        xlabel('Real','interpreter','latex')
        ylabel('Imag','interpreter','latex')
        saveas(gcf,['Ex1_AR_Eig',num2str(n),'_A.pdf']);
        
        figure('visible','off');
        hold on
        plot(real(eyac),imag(eyac),'x');
        hold off
        set(gca,'fontsize',20);
        set(gca,'TickLabelInterpreter','latex')
        xlabel('Real','interpreter','latex')
        ylabel('Imag','interpreter','latex')
        saveas(gcf,['Ex1_AR_Eig',num2str(n),'_YA.pdf']);
        save Ex1_AR_Eig eac eyac
    end
end
fclose(fid);
end


function y = tmtimes(Tsmtmult,x,transp_flag)
% Toeplitz matrix-vector multiplication
if strcmp(transp_flag,'transp')
    y = mtimes(x',Tsmtmult)';
else
    y = mtimes(Tsmtmult,x);
end
end

function y = ymtimes(Tstmult,x)
% Flipped Toeplitz matrix-vector multiplication
y = mtimes(Tstmult,x);
y = y(end:-1:1);
end

function y = capply(Csmt,x)
y = mldivide(Csmt,x);
end

function y = capplytrans(Csmt,Csmttrans,x,transp_flag)
% V-cycle for LSQR
if strcmp(transp_flag,'transp')
    y = mldivide(Csmttrans,x);
else
    y = mldivide(Csmt,x);
end
end

