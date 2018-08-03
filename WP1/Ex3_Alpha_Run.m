function [t,it,rv,dof,runt] = Ex3_Alpha_Run

% [t,it,rv,dof,runt] = Ex3_Alpha_Run
% 
% Example 3, Table 5.6 from J. Pestana, Preconditioners for symmetrized 
% Toeplitz and multilevel Toeplitz matrices, 2018.
%
% Outputs:  t: array of CPU times
%           it: array of iterationa
%           rv: array of relative residuals
%
% J. Pestana, August 3, 2018

addpath(genpath('../smt'));
addpath(genpath('../sptoeplitz'));

% Problem specs
R = [1,1];      % Right end of spatial interval
L = [0,0];      % Left end of spatial interval
T = 1;          % Right end of time interval
dv = [2 0.5];   % Diffusion coeffs in x direction
ev = [0.3 1];   % Diffusion coeffs in y direction

% List of fractional diffusion orders
fraclist = [1.5 1.25; 1.5 1.75];


gvec = (2.^(5:2:9))-1; % Problem dimensions
mtstep = 1;            % Number of time steps

n_time = gvec(end);
n_grid = length(gvec);
n_type = 6;
n_fracs = size(fraclist,1);

% Store results
t(n_grid,1:n_type,n_fracs) = 0;
it(n_grid,n_time,1:n_type,n_fracs) = 0;
rv(n_grid,1:n_type,n_fracs) = 0;
dof(n_grid,n_fracs) = 0;
runt(n_grid,1) = 0;

tol = 1e-8; % Solver tolerance

% Loop over problem dimensions
for gr = 1:n_grid 
    N = [gvec(gr) gvec(gr)]; % Grid in x and y directions
    num_levels = log2(gvec(gr)+1);
    mg_levels = num_levels - 3; % Number of multigrid levels
    
    NN = N(1)*N(2);
    x0 = ones(NN,1)/sqrt(NN); % Initial guess
        
    % Loop over alpha
    for k = 1:n_fracs
        alph = fraclist(k,1);       % frac diff power in x direction
        beta = fraclist(k,2);       % frac diff power in y direction
        M = ceil(N(1)^alph);        % Total number of time steps

        fprintf('N1= %i, N2 = %i, M = %i, alpha = %g, beta = %g\n',N,M,alph,beta);
        
        % Build matrices
        fprintf('Building matrices ...')
        [Axsmt,Aysmt,u0,F] = Ex3_Gen_Toep(L,R,T,N,M,mtstep,alph,beta,dv,ev);
        Axsmtmult = toeprem(Axsmt);
        Aysmtmult = toeprem(Aysmt);
        dof(gr,n_fracs) = size(F,1)*size(F,2);
        fprintf('Done\n');
        
        

        
        % Right pre GMRES with circulant preconditioner
        fprintf('GMRES Circ ...')
        U1 = zeros(NN,mtstep);
        u_old = u0;
        tic;
        Cx = smtcprec('strang',Axsmt);  % 1D circulant: x direction
        eCx = fft(Cx(:,1));             % Eigenvalues of Cx
        Cy = smtcprec('strang',Aysmt);  % 1D circulant: y direction
        eCy = fft(Cy(:,1));             % Eigenvalues of Cy
        
        for ts = 1:mtstep
            b = F(:,ts) + u_old;
            [u_old,~,itgp] = rpgmres(@(x)matvecmult(Axsmtmult,Aysmtmult,N,x),b,tol,80,1,@(x)applyP1(eCx,eCy,N,x,'notransp'),x0);
            U1(:,ts) = u_old;
            it(gr,ts,1,k) = itgp(2);
        end
        t(gr,1,k) = toc;
        rv(gr,1,k) = norm(matvecmult(Axsmtmult,Aysmtmult,N,U1(:,1)) - (F(:,1)+u0),'fro');
        fprintf('done\n');
         
        % Right pre GMRES with MG preconditioner
        fprintf('GMRES MG...')
        U2 = zeros(NN,mtstep);
        u_old = u0;
        tic;
        IAx = full(eye(N(1))/2 - Axsmt);
        IAy = full(eye(N(1))/2 - Aysmt);
        [diagel,Lm,IIm,IAxm,IAymt] = vcycle_2d_setup(IAx,IAy,mg_levels);
        for ts = 1:mtstep
            b = F(:,ts) + u_old;
            [u_old,~,itgp] = rpgmres(@(x)matvecmult(Axsmtmult,Aysmtmult,N,x),b,tol,20,1,@(y)vcycle_2d(IAxm,IAymt,IIm,diagel,Lm,y,4,4,1,mg_levels,0.9),x0);
            U2(:,ts) = u_old;
            it(gr,ts,2,k) = itgp(2);
        end
        t(gr,2,k) = toc;
        rv(gr,2,k) = norm(matvecmult(Axsmtmult,Aysmtmult,N,U2(:,1)) - (F(:,1)+u0),'fro');
        fprintf('done\n');
         
        % LSQR with circulant preconditioner
        fprintf('LSQR Circ...')
        U3 = zeros(NN,mtstep);
        u_old = u0;
        tic;
        Cx = smtcprec('strang',Axsmt);  % 1D circulant: x direction
        eCx = fft(Cx(:,1));             % Eigenvalues of Cx
        Cy = smtcprec('strang',Aysmt);  % 1D circulant: y direction
        eCy = fft(Cy(:,1));             % Eigenvalues of Cy
        for ts = 1:mtstep
            b = F(:,ts) + u_old;
            [u_old,~,~,itls] = lsqr(@(x,transp_flag)qmrmtimes(Axsmtmult,Aysmtmult,N,x,transp_flag),b,tol,1000,@(x,transp_flag)applyP1(eCx,eCy,N,x,transp_flag));
            U3(:,ts) = u_old;
            it(gr,ts,3,k) = itls;
        end
        t(gr,3,k) = toc;
        rv(gr,3,k) = norm(matvecmult(Axsmtmult,Aysmtmult,N,U3(:,1)) - (F(:,1)+u0),'fro');
        fprintf('done\n');
        
        % LSQR with MG preconditioner
        fprintf('LSQR MG...')
        U4 = zeros(NN,mtstep);
        u_old = u0;
        tic;
        IAx = full(eye(N(1))/2 - Axsmt);
        IAy = full(eye(N(1))/2 - Aysmt);
        
        [diagel,Lm,IIm,IAxm,IAym] = vcycle_2d_setup(IAx,IAy,mg_levels);
        [diageltrans,Lmtrans,IImtrans,IAxmtrans,IAymtrans] = vcycle_2d_setup(IAx',IAy',mg_levels);
        for ts = 1:mtstep
            b = F(:,ts) + u_old;
            [u_old,~,~,itls] = lsqr(@(x,transp_flag)qmrmtimes(Axsmtmult,Aysmtmult,N,x,transp_flag),b,tol,1000,@(y,transp_flag)mglsqr(y,0.9,diagel,IAxm,IAym,IIm,Lm,diageltrans,IAxmtrans,IAymtrans,IImtrans,Lmtrans,mg_levels,transp_flag));
            U4(:,ts) = u_old;
            it(gr,ts,4,k) = itls;
        end
        t(gr,4,k) = toc;
        rv(gr,4,k) = norm(matvecmult(Axsmtmult,Aysmtmult,N,U4(:,1)) - (F(:,1)+u0),'fro');
        fprintf('done\n');       
        
        % MINRES with symmetrized pos def preconditioner
        fprintf('MINRES Circ ...')
        u_old = u0;
        U5 = zeros(NN,mtstep);
        tic;
        Cx = smtcprec('strang',Axsmt); Cpx = smcirc(ifft(abs(fft(Cx(:,1)))));
        Cy = smtcprec('strang',Aysmt); Cpy = smcirc(ifft(abs(fft(Cy(:,1)))));
        eCpx = fft(Cpx(:,1)); eCpy = fft(Cpy(:,1));
        for ts = 1:mtstep
            yb = flipud(F(:,ts) + u_old);
            [u_old,~,~,itmr] = minres(@(x)ymtimes(Axsmtmult,Aysmtmult,N,x),yb,tol,1000,@(x)applyP1pos(eCpx,eCpy,N,x),[],x0);
            U5(:,ts) = u_old;
            it(gr,ts,5,k) = itmr;
        end
        t(gr,5,k) = toc;
        rv(gr,5,k) = norm(matvecmult(Axsmtmult,Aysmtmult,N,U5(:,1)) - (F(:,1)+u0),'fro');
        fprintf('done\n');
        
        % MINRES with MG preconditioner
        fprintf('MINRES MG...')
        u_old = u0;
        U6 = zeros(NN,mtstep);
        tic;
        IAx = full(eye(N(1))/2 - (Axsmt+Axsmt')/2);
        IAy = full(eye(N(1))/2 - (Aysmt+Aysmt')/2);
        [diagel,Lm,IIm,IAxm,IAym] = vcycle_2d_setup(IAx,IAy,mg_levels);
        for ts = 1:mtstep
            yb = flipud(F(:,ts) + u_old);
            [u_old,~,~,itmr] = minres(@(x)ymtimes(Axsmtmult,Aysmtmult,N,x),yb,tol,1000,@(y)vcycle_2d(IAxm,IAym,IIm,diagel,Lm,y,4,4,1,mg_levels,0.9),[],x0);
            U6(:,ts) = u_old;
            it(gr,ts,6,k) = itmr;
        end
        t(gr,6,k) = toc;
        rv(gr,6,k) = norm(matvecmult(Axsmtmult,Aysmtmult,N,U6(:,1)) - (F(:,1)+u0),'fro');
        fprintf('done\n');
        
        
        it(:,mtstep+1:end,:,:) = [];
        save Ex3_Alpha t it rv runt dof gvec
        
    end
end

% Make table
fid = fopen('Ex3_Alpha.txt','w+');
for k=1:n_fracs
    fprintf(fid,'\\hline\n');
    fprintf(fid,'\\multirow{%i}{*}{(%g,%g)} ',n_grid,fraclist(k,:));
    for j = 1:n_grid
        fprintf(fid,'& %i ',gvec(j)^2);
        fprintf(fid,'& %i & (%3.2g) & %i & (%3.2g)& %i & (%3.2g)',max(it(j,:,1,k)),t(j,1,k)/mtstep,max(it(j,:,2,k)),t(j,2,k)/mtstep,max(it(j,:,3,k)),t(j,3,k)/mtstep);
        fprintf(fid,'& %i & (%3.2g) & %i & (%3.2g) & %i & (%3.2g)',max(it(j,:,4,k)),t(j,4,k)/mtstep,max(it(j,:,5,k)),t(j,5,k)/mtstep,max(it(j,:,6,k)),t(j,6,k)/mtstep);
        fprintf(fid,'\\\\\n');
    end
end
fclose all;

end

% Matrix-vector product with Toeplitz matrix
function y = matvecmult(Axsmtmult,Aysmtmult,N,x)
X = reshape(x,N(2),N(1));
y = reshape(X - mtimes(Axsmtmult,X)  - mtimes(Aysmtmult,X')',N(1)*N(2),1);
end


% Matrix-vector product with Toeplitz matrix (for LSQR)
function y = qmrmtimes(Axsmtmult,Aysmtmult,N,x,transp_flag)

if strcmp(transp_flag,'transp')
    X = reshape(x,N(2),N(1));
    y = reshape(X - mtimes(Axsmtmult',X)  - mtimes(Aysmtmult',X')',N(2)*N(1),1);
else
    X = reshape(x,N(2),N(1));
    y = reshape(X - mtimes(Axsmtmult,X)  - mtimes(Aysmtmult,X')',N(2)*N(1),1);
end
end

% Matrix-vector product with flipped Toeplitz matrix
function y = ymtimes(Axsmtmult,Aysmtmult,N,x)
X = reshape(x,N(2),N(1));
y = flipud(reshape(X - mtimes(Axsmtmult,X)  - mtimes(Aysmtmult,X')',N(1)*N(2),1));
end


% Apply circulant preconditioner
function y = applyP1(eCx,eCy,N,x,transp_flag)
X = reshape(x,N(2),N(1));

U = ifft(ifft(X')');
if strcmp(transp_flag,'transp')
    v = ones(N(1)*N(2),1) - repmat(conj(eCx),N(2),1) - kron(conj(eCy),ones(N(1),1));
else
    v = ones(N(1)*N(2),1) - repmat(eCx,N(2),1) - kron(eCy,ones(N(1),1));
end
W = U./reshape(v,N(2),N(1));
Y = fft(fft((reshape(W,N(2),N(1)))')');
y = reshape(Y,N(1)*N(2),1);
end

% Apply positive definite circulant preconditioner
function y = applyP1pos(eCpx,eCpy,N,x)
X = reshape(x,N(2),N(1));

U = ifft(ifft(X')');
v = ones(N(1)*N(2),1) + repmat(eCpx,N(2),1) + kron(eCpy,ones(N(1),1));
W = U./reshape(v,N(2),N(1));
Y = fft(fft((reshape(W,N(2),N(1)))')');
y = reshape(Y,N(1)*N(2),1);
end

% Multigrid for LSQR
function y = mglsqr(y,omega,diagel,IAxm,IAym,IIm,Lm,diagelt,IAxmt,IAymt,IImt,Lmt,mg_levels,transp_flag)

if strcmp(transp_flag,'transp')
    y = vcycle_2d(IAxmt,IAymt,IImt,diagelt,Lmt,y,4,4,1,mg_levels,omega);
else
    y = vcycle_2d(IAxm,IAym,IIm,diagel,Lm,y,4,4,1,mg_levels,omega);
end
end