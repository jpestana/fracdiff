function [iv,tv,rv] = Ex1_AF_Run

% [it,tv,rv] = Ex1_AF_Run
%
% Example 1, Table 5.2 from J. Pestana, Preconditioners for symmetrized 
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

fid = fopen('Ex1_AF.txt','w+'); % Open file for table
eigcalc = 0;  % 1 for eigenvalue computation, 0 otherwise

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
    lev = log2(n+1);
    mglev = lev-4;
    disp(n)
    
    fprintf('Building matrix...')
    load(['Ex1_cr_',num2str(n),'.mat'],'c','r');
    r(1) = c(1);                    % Consistency requirement
    Tsmt = smtoep(c,r);
    Tsmtmult = toeprem(Tsmt);
    ca = c; ra = r;
    fprintf('Done\n');
    
    fprintf('Getting preconditioner...')
    [c,r] = Ex1_Gen_Toep_Abs_F(n);
    r(1) = c(1);                    % Consistency requirement
    P = toeplitz(c,r);
    fprintf('Done\n');
    
    b = randn(n,1); b = b/norm(b);  % RHS             
    tol = 1e-8;                     % Iterative solver tolerance
    x0 = ones(n,1)/sqrt(n);         % Intial guess
    
    % GMRES with exact pre
    fprintf('GMRES with exact pre...')
    tic;
    [~, ~, itgpc, ~,resgpc] = rpgmres( @(x)tmtimes(Tsmtmult,x,'notransp'), b, tol, n, 1, full(Tsmt), x0 );
    tv(j,1) = toc;
    rv(j,1) = resgpc(end);
    iv(j,1) = itgpc(2);
    fprintf('Done\n');
    
    % GMRES with multigrid preconditioner
    fprintf('GMRES with multigrid preconditioner...')
    tic;
    [diagel,Asm,Ac] = vcycle_fastmv_setup(ca,ra,mglev);
    [~, ~, itgp, ~,resgp] = rpgmres( @(x)tmtimes(Tsmtmult,x,'notransp'), b, tol, n, 1,@(y)vcycle_fastmv_nr(diagel,Asm,Ac,y,2,2,mglev,0.1), x0 );
    tv(j,2) = toc;
    rv(j,2) = resgp(end);
    iv(j,2) = itgp(2);
    fprintf('Done\n');
    
    % LSQR with exact preconditioner
    fprintf('LSQR with exact preconditioner...')
    tic;
    [~, ~, ~,iv(j,3), reslpc] = lsqr(@(x,transp_flag)tmtimes(Tsmtmult,x,transp_flag),b,tol,n,full(Tsmt),[],x0);
    tv(j,3) = toc;
    rv(j,3) = reslpc(end);
    fprintf('Done\n');
    
    % LSQR with multigrid preconditioner
    fprintf('LSQR with multigrid preconditioner...')
    tic;
    [diagel,Asm,Ac] = vcycle_fastmv_setup(c,r,mglev);
    [diagelt,Asmt,Act] = vcycle_fastmv_setup(conj(r),conj(c),mglev);
    [~, ~, ~,iv(j,4), reslp] = lsqr(@(x,transp_flag)tmtimes(Tsmtmult,x,transp_flag),b,tol,n,@(y,transp_flag)vcyclelsqr(y,diagel,diagelt,Asm,Asmt,Ac,Act,2,2,mglev,0.4,transp_flag),[],x0);
    tv(j,4) = toc;
    rv(j,4) = reslp(end);
    fprintf('Done\n');
    
    % MINRES with exact preconditioner
    fprintf('MINRES with exact preconditioner...')
    tic;
    Yb = b(n:-1:1);                 % Apply Y to b
    [~,~,~,iv(j,5),resmpc] = minres(@(x)ymtimes(Tsmtmult,x),Yb,tol,n,P,[],x0);
    tv(j,5) = toc;
    rv(j,5) = resmpc(end);
    fprintf('Done\n');
    
    % MINRES with multigrid preconditioner
    fprintf('MINRES with multigrid preconditioner...')
    tic;
    Yb = b(n:-1:1);                 % Apply Y to b
    [diagel,Asm,Ac] = vcycle_fastmv_setup(c,r,mglev);
    [~,~,~,iv(j,6),resmp] = minres(@(x)ymtimes(Tsmtmult,x),Yb,tol,n,@(y)vcycle_fastmv_nr(diagel,Asm,Ac,y,2,2,mglev,0.5),[],x0);
    tv(j,6) = toc;
    rv(j,6) = resmp(end);
    fprintf('Done\n');
    save Ex1_AF iv tv rv
    
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
        saveas(gcf,['Ex1_AF_Eig',num2str(n),'_A.pdf']);
        
        figure('visible','off');
        hold on
        plot(real(eyac),imag(eyac),'x');
        hold off
        set(gca,'fontsize',20);
        set(gca,'TickLabelInterpreter','latex')
        xlabel('Real','interpreter','latex')
        ylabel('Imag','interpreter','latex')
        saveas(gcf,['Ex1_AF_Eig',num2str(n),'_YA.pdf']);
        save Ex1_AF_Eig eac eyac
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

function u = vcyclelsqr(y,diagel,diagelt,Asm,Asmt,Ac,Act,numpre,numpost,numlev,om,transp_flag)
% V-cycle for LSQR
if strcmp(transp_flag,'transp')
    u = vcycle_fastmv_nr(diagelt,Asmt,Act,y,numpre,numpost,numlev,om);
else
    u = vcycle_fastmv_nr(diagel,Asm,Ac,y,numpre,numpost,numlev,om);
end
end

