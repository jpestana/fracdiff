function [t,it,rv] = Ex2_Const_Nu_Alpha_Run

% function [t,it,rv] = Ex2_Const_Nu_Alpha_Run
%
% Example 2, Table 5.3 from J. Pestana, Preconditioners for symmetrized 
% Toeplitz and multilevel Toeplitz matrices, 2018.
%
% Outputs:  t: array of CPU times
%           it: array of iterations
%           rv: array of relative residuals
%
% J. Pestana, July 26, 2018


addpath(genpath('../smt'));

prob = 3;

if prob == 1 || prob == 2
    % Problem specs
    R = 2; % Right end of spatial interval
    L = 0; % Left end of spatial interval
    T = 1; % Right end of time interval
elseif prob == 3 || prob == 4
    % BSS
    R = 1; % Right end of spatial interval
    L = 0; % Left end of spatial interval
    T = 1; % Right end of time interval
end

% List of fractional diffusion orders
alplist = [1.25 1.5 1.75];

% Diffusion coefficients
if prob == 1
    d1 = 0.4;
    d2 = 0.8;
elseif prob == 2
    d1 = 0.2;
    d2 = 1;
elseif prob == 3 || prob == 4
    d1 = 0.5;
    d2 = 1;
end

fprintf('d1 = %g, d2 = %g\n',d1,d2);

gvec = (2.^(10:2:18))-1; % Problem dimensions
mtstep = 1;              % Number of time steps

n_grid = length(gvec);
n_time = mtstep;
n_alpha = length(alplist);
n_type = 7;

% Store results
t(n_grid,1:n_type,n_alpha) = 0;
it(n_grid,n_time,1:n_type,n_alpha) = 0;
rv(n_grid,1:n_type,n_alpha) = 0;
runt(n_grid,1) = 0;

tol = 1e-8; % Solver tolerance

% Loop over problem dimensions
for gr = 1:n_grid
    N = gvec(gr);
    lev = log2(N+1);
    stoplev = lev-7; % Number of levels for multigrid
    x0 = ones(N,1)/sqrt(N); % Starting guess
    
    % Loop over alpha
    for k = 1:n_alpha
        alpha = alplist(k);
        M = ceil(N^alpha);
        
        fprintf('N= %i, M = %i, alpha = %g\n',N,M,alpha);
        
        % Build matrices
        fprintf('Building matrices ...')
        [Asmt,u0,F,dx,nu,col,row] = Ex2_Gen_Toep(L,R,T,N,M,mtstep,d1,d2,alpha,prob);
        Fdx = dx^alpha*F;
        Asmtmult = toeprem(Asmt);
        fprintf('Done\n');
        
        nb = norm(Fdx,'fro');
        
        % Right pre GMRES with circulant preconditioner
        fprintf('GMRES Circ ...')
        U1 = zeros(N,mtstep);
        u_old = u0;
        tic;
        c = zeros(N,1);
        mi = floor(N/2);
        c(1:mi+1) = Asmt.t(N:N+mi);
        c(mi+2:N) = Asmt.t(mi+1:N-1);
        laminv = 1./fft(c);
        for ts = 1:mtstep
            b = Fdx(:,ts) + nu*u_old;
            [u_old,~,itgp] = rpgmres(@(x)mtimes(Asmtmult,x),b,tol,20,1,@(x)mycircinv(laminv,x),x0);
            U1(:,ts) = u_old;
            it(gr,ts,1,k) = itgp(2);
        end
        t(gr,1,k) = toc;
        rv(gr,1,k) = norm(Fdx + nu*[u0 U1(:,1:end-1)] - mtimes(Asmtmult,U1),'fro')/nb;
        fprintf('done\n');
        
        % Right pre GMRES with MG
        fprintf('GMRES MG...')
        U2 = zeros(N,mtstep);
        u_old = u0;
        tic;
        [diagel,Asm,Ac] = vcycle_fastmv_setup(col',row',stoplev);
        for ts = 1:mtstep 
            b = Fdx(:,ts) + nu*u_old;
            [u_old,~,itgp] = rpgmres(@(x)mtimes(Asmtmult,x),b,tol,20,1,@(y)vcycle_fastmv_nr(diagel,Asm,Ac,y,2,2,stoplev,0.7),x0);
            U2(:,ts) = u_old;
            it(gr,ts,2,k) = itgp(2);
        end
        t(gr,2,k) = toc;
        rv(gr,2,k) = norm(Fdx + nu*[u0 U2(:,1:end-1)] - mtimes(Asmtmult,U2),'fro')/nb;
        fprintf('done\n');
        
        % LSQR with circulant preconditioner
        fprintf('LSQR Circ ...')
        U3 = zeros(N,mtstep);
        u_old = u0;
        
        tic;
        c = zeros(N,1);
        mi = floor(N/2);
        c(1:mi+1) = Asmt.t(N:N+mi);
        c(mi+2:N) = Asmt.t(mi+1:N-1);
        laminv = 1./fft(c);
        for ts = 1:mtstep
            b = Fdx(:,ts) + nu*u_old;
            [u_old,~,~,itls] = lsqr(@(x,transp_flag)qmrmtimes(Asmtmult,x,transp_flag),b,tol,N,@(x,transp_flag)qmrmldivide(laminv,conj(laminv),x,transp_flag),[],x0);
            U3(:,ts) = u_old;
            it(gr,ts,3,k) = itls;
        end
        t(gr,3,k) = toc;
        rv(gr,3,k) = norm(Fdx + nu*[u0 U3(:,1:end-1)] - mtimes(Asmtmult,U3),'fro')/nb;
        fprintf('done\n');
        
        % LSQR with MG
        fprintf('LSQR MG ...')
        U4 = zeros(N,mtstep);
        u_old = u0;
        tic;
        [diagel,Asm,Ac] = vcycle_fastmv_setup(col',row',stoplev);
        [diageltrans,Asmtrans,Actrans] = vcycle_fastmv_setup(row',col',stoplev);
        for ts = 1:mtstep
            b = Fdx(:,ts) + nu*u_old;
            [u_old,~,~,itls] = lsqr(@(x,transp_flag)qmrmtimes(Asmtmult,x,transp_flag),b,tol,N,@(y,transp_flag)mglsqr(y,diagel,Asm,Ac,diageltrans,Asmtrans,Actrans,stoplev,transp_flag),[],x0);
            U4(:,ts) = u_old;
            it(gr,ts,4,k) = itls;
        end
        t(gr,4,k) = toc;
        rv(gr,4,k) = norm(Fdx + nu*[u0 U4(:,1:end-1)] - mtimes(Asmtmult,U4),'fro')/nb;
        fprintf('done\n');
        
        % MINRES with circulant
        fprintf('MINRES Circ ...')
        u_old = u0;
        U5 = zeros(N,mtstep);
        tic;
        c = zeros(N,1);
        mi = floor(N/2);
        c(1:mi+1) = Asmt.t(N:N+mi);
        c(mi+2:N) = Asmt.t(mi+1:N-1);
        laminv = 1./abs(fft(c));
        for ts = 1:mtstep
            yb = flipud(Fdx(:,ts) + nu*u_old);
            [u_old,~,~,itmr] = minres(@(x)ymtimes(Asmtmult,x),yb,tol,N,@(x)mycircinv(laminv,x),[],x0);
            U5(:,ts) = u_old;
            it(gr,ts,5,k) = itmr;
        end
        t(gr,5,k) = toc;
        rv(gr,5,k) = norm(Fdx + nu*[u0 U5(:,1:end-1)] - mtimes(Asmtmult,U5),'fro')/nb;
        fprintf('done\n');
        
         % MINRES with MG AR
        fprintf('MINRES MG AR...')
        u_old = u0;
        U7 = zeros(N,mtstep);
        tic;
        rowcol = (row + col)'/2;
        [diagel,Asm,Ac] = vcycle_fastmv_setup(rowcol,rowcol,stoplev);
        for ts = 1:mtstep
            yb = flipud(Fdx(:,ts) + nu*u_old);
            [u_old,~,~,itmr] = minres(@(x)ymtimes(Asmtmult,x),yb,tol,N,@(y)vcycle_fastmv_nr(diagel,Asm,Ac,y,2,2,stoplev,0.7),[],x0);
            U7(:,ts) = u_old;
            it(gr,ts,7,k) = itmr;
        end
        t(gr,7,k) = toc;
        rv(gr,7,k) = norm(Fdx + nu*[u0 U7(:,1:end-1)] - mtimes(Asmtmult,U7),'fro')/nb;
        fprintf('done\n');
        
        % MINRES with MG AF
        fprintf('MINRES MG AF...')
        u_old = u0;
        U6 = zeros(N,mtstep);
        tic;
        if alpha <= 1.4
            [colaf,rowaf] = Ex2GenToep_AF(N,nu,alpha,d1,d2,ceil(50));
        elseif alpha >1.4 && alpha < 1.6
            [colaf,rowaf] = Ex2GenToep_AF(N,nu,alpha,d1,d2,ceil(40*1.1^(lev)));
        else
            [colaf,rowaf] = Ex2GenToep_AF(N,nu,alpha,d1,d2,ceil(100*1.1^lev));
        end
        [diagel,Asm,Ac] = vcycle_fastmv_setup(colaf,rowaf,stoplev);
        for ts = 1:mtstep
            yb = flipud(Fdx(:,ts) + nu*u_old);
            [u_old,~,~,itmr] = minres(@(x)ymtimes(Asmtmult,x),yb,tol,N,@(y)vcycle_fastmv_nr(diagel,Asm,Ac,y,1,1,stoplev,0.7),[],x0);
            U6(:,ts) = u_old;
            it(gr,ts,6,k) = itmr;
        end
        t(gr,6,k) = toc;
        rv(gr,6,k) = norm(Fdx + nu*[u0 U6(:,1:end-1)] - mtimes(Asmtmult,U6),'fro')/nb;
        fprintf('done\n');
        
        save Ex2_Const_Nu_Alpha t it rv runt gvec d1 d2 L R T alplist mtstep prob
    end
    
end

% Make table
fid = fopen('Ex2_Const_Nu_Alpha.txt','w+');
for k=1:n_alpha
    fprintf(fid,'\\hline\n');
    fprintf(fid,'\\multirow{%i}{*}{%g} ',n_grid,alplist(k));
    for j = 1:n_grid
        fprintf(fid,'& %i ',gvec(j));
        fprintf(fid,'& %i & (%3.2g) & %i & (%3.2g)',max(it(j,:,1,k)),t(j,1,k)/mtstep,max(it(j,:,2,k)),t(j,2,k)/mtstep);
        fprintf(fid,'& %i & (%3.2g) & %i & (%3.2g)',max(it(j,:,3,k)),t(j,3,k)/mtstep,max(it(j,:,4,k)),t(j,4,k)/mtstep);
        fprintf(fid,'& %i & (%3.2g) & %i & (%3.2g)',max(it(j,:,5,k)),t(j,5,k)/mtstep,max(it(j,:,6,k)),t(j,6,k)/mtstep);
        fprintf(fid,'& %i & (%3.2g)\\\\\n', max(it(j,:,7,k)),t(j,7,k)/mtstep);
    end
end
fclose all;

end

% Matrix-vector product with Toeplitz matrix
function y = qmrmtimes(Tsmtmult,x,transp_flag)
if strcmp(transp_flag,'transp')
    y = mtimes(x',Tsmtmult)';
else
    y = mtimes(Tsmtmult,x);
end
end

% Circulant preconditioner solve for LSQR
function y = qmrmldivide(laminv,laminvtrans,x,transp_flag)
if strcmp(transp_flag,'transp')
    y = laminvtrans.*fft(x);
else
    y = laminv.*fft(x);
end
if abs(imag(y))<1e-12
    y = real(y);
end
y = real(ifft(y));
end

% Circulant preconditioner solve
function y = mycircinv(laminv,x)
y = laminv.*fft(x);
if(abs(imag(y)))<1e-12
    y = real(y);
end
y = real(ifft(y));

end

% Multigrid for LSQR
function y = mglsqr(y,diagel,Asm,Ac,diageltrans,Asmtrans,Actrans,stoplev,transp_flag)
if strcmp(transp_flag,'transp')
    y = vcycle_fastmv_nr(diageltrans,Asmtrans,Actrans,y,1,1,stoplev,0.7);
else
    y = vcycle_fastmv_nr(diagel,Asm,Ac,y,1,1,stoplev,0.7);
end
end

% Matrix-vector product with flipped Toeplitz matrix
function y = ymtimes(Tstmult,x)
y = mtimes(Tstmult,x);
y = y(end:-1:1);
end