%function test_driver_general
clc
clear all
addpath ../poblano_toolbox
addpath ../poblano_toolbox_ext
figStart=0;
%------------------------------
% Parameters
%------------------------------
%Shared parameters (for NGMRES, NCG, LBFGS)
params.shared.MaxIters = 500;
params.shared.MaxFuncEvals = 2000;
params.shared.RelFuncTol =-1;
params.shared.GNormTol = 0;
params.shared.TraceFuncEvals = true;
params.shared.TraceFunc = true;
params.shared.TraceRelFunc = true;
params.shared.TraceGradNorm = true;

%Linesearch parameters
params.ls.LineSearch_ftol = 1e-4;
params.ls.LineSearch_gtol = 1e-2;
params.ls.LineSearch_initialstep = 1;
params.ls.LineSearch_maxfev = 20;
params.ls.LineSearch_method = 'more-thuente';
params.ls.LineSearch_stpmax = 1e15;
params.ls.LineSearch_stpmin = 1e-15;
params.ls.LineSearch_xtol = 1e-15;

% NGMRES parameters
params.ngmres = ngmres_prec('defaults');
params.ngmres.W = 20; % Maximum window size
params.ngmres.TraceRestarts = false;  % Flag to include the history of the restarts
params.ngmres.Display = 'off';  % Options: 'iter', 'final' or 'off'
params.ngmres.UsePrecond = false;
params.ngmres.LineSearch_method = 'more-thuente';
params.ngmres.NPrecond = 1;

% NCG parameters (for comparing NCG with NGMRES)
params.ncg = ncg_prec('defaults');
params.ncg.RestartNW = false;
params.ncg.Update = 'PR'; % 'HS'
params.ncg.Display = 'off'; % Options: 'iter', 'final' or 'off'
params.ncg.UsePrecond = false;
params.ncg.LineSearch_method = 'more-thuente';

% LBFGS parameters (for comparing LBFGS with NGMRES)
params.lbfgs = lbfgs_prec('defaults');
params.lbfgs.M = 5; % Window size
params.lbfgs.Display = 'off'; % Options: 'iter', 'final' or 'off'
params.lbfgs.LineSearch_method = 'more-thuente';

% Nesterov parameters
params.nesterov = nesterov_prec('defaults');
params.nesterov.Display = 'off';  % Options: 'iter', 'final' or 'off'
params.nesterov.UsePrecond = false;
params.nesterov.NoPrecondStep = 'ls'; % if above is false. Options: 'ls','fixed'
params.nesterov.eta = 1; % value of eta
params.nesterov.alpha = 2.3*1e-5; % value of alpha (fixed gradient descent step)
params.nesterov.delay = 1; % 1: no delay
params.nesterov.step_type = 2; % 1: step 1; 2: gradient ratio; 3: Nesterov sequence; 4: line search on beta
params.nesterov.restart_type = 2; % 0: no restart; 1: function; 2: gradient; 3: speed (x-based)

%Load Shared Parameters;
params = test_SetSharedParameters(params);

%------------------------------
% General parameters
%------------------------------
params.initSeed = 0;
rs = RandStream('mt19937ar','Seed',params.initSeed);
RandStream.setGlobalStream(rs);

%--------------------------------------
% Set up the test problem
%--------------------------------------
% standard quadratic function with diagonal matrix A
n = 100;
kappa = 1;
d = (1:n)';
d(n) = d(n)*kappa;
x0 = rand(n,1); % generate the initial guess
x_exact = ones(n,1);
alpha = 10; % factor in the paraboloid coordinate transformation
fg = @(x) test_objective(x,d,x_exact,alpha); % function and gradient

%--------------------------------------
% Call the Nesterov method
%--------------------------------------
fprintf('+++ Start Nesterov +++\n');
nesterovtime = tic;
out_nesterov = nesterov_prec(fg, x0, params.nesterov);
out_nesterov.time = toc(nesterovtime);
for i = 2:size(out_nesterov.TraceFuncEvals,2)
    out_nesterov.TraceFuncEvals(i) = out_nesterov.TraceFuncEvals(i)+out_nesterov.TraceFuncEvals(i-1);
end
fprintf(' Iter   Time   FuncEvals       F(X)          ||G(X)||/N        \n');
fprintf('------  -----  --------- ---------------- ----------------\n');
fprintf('%6d %5.2g %9d %16.8g %16.8g\n', out_nesterov.Iters,...
    out_nesterov.time, round(out_nesterov.FuncEvals), ...
    out_nesterov.F, norm(out_nesterov.G)/n);

%--------------------------------------
% Call LBFGS
%--------------------------------------
fprintf('+++ Start LBFGS +++\n');
lbfgstime = tic;
out_lbfgs = lbfgs_prec(fg,x0,params.lbfgs);
out_lbfgs.time = toc(lbfgstime);
for i=2:size(out_lbfgs.TraceFuncEvals,2)
    out_lbfgs.TraceFuncEvals(i)= out_lbfgs.TraceFuncEvals(i)+...
        out_lbfgs.TraceFuncEvals(i-1);
end
fprintf(' Iter   Time   FuncEvals       F(X)          ||G(X)||/N        \n');
fprintf('------  -----  --------- ---------------- ----------------\n');
fprintf('%6d %5.2g %9d %16.8g %16.8g\n', out_lbfgs.Iters,...
    out_lbfgs.time, round(out_lbfgs.FuncEvals), ...
    out_lbfgs.F, norm(out_lbfgs.G)/n);

%--------------------------------------
% Call the NGMRES method
%--------------------------------------
fprintf('+++ Start N-GMRES - Steepest Descent with LineSearch +++\n');
ngmrestime = tic;
params.ngmres.NoPrecondStep = 'ls';
out_ngmres_ls = ngmres_prec(fg, x0, params.ngmres);
out_ngmres_ls.time = toc(ngmrestime);
for i = 2:size(out_ngmres_ls.TraceFuncEvals,2)
    out_ngmres_ls.TraceFuncEvals(i) = out_ngmres_ls.TraceFuncEvals(i)...
        +out_ngmres_ls.TraceFuncEvals(i-1);
end
fprintf(' Iter   Time   FuncEvals       F(X)          ||G(X)||/N        \n');
fprintf('------  -----  --------- ---------------- ----------------\n');
fprintf('%6d %5.2g %9d %16.8g %16.8g\n', out_ngmres_ls.Iters,...
    out_ngmres_ls.time, round(out_ngmres_ls.FuncEvals), ...
    out_ngmres_ls.F, norm(out_ngmres_ls.G)/n);

fprintf('+++ Start N-GMRES - Steepest Descent with Fixed Step +++\n');
ngmrestime = tic;
params.ngmres.NoPrecondStep = 'fixed';
out_ngmres_fix = ngmres_prec(fg, x0, params.ngmres);
out_ngmres_fix.time = toc(ngmrestime);
for i = 2:size(out_ngmres_fix.TraceFuncEvals,2)
    out_ngmres_fix.TraceFuncEvals(i) = out_ngmres_fix.TraceFuncEvals(i)...
        +out_ngmres_fix.TraceFuncEvals(i-1);
end
fprintf(' Iter   Time   FuncEvals       F(X)          ||G(X)||/N        \n');
fprintf('------  -----  --------- ---------------- ----------------\n');
fprintf('%6d %5.2g %9d %16.8g %16.8g\n', out_ngmres_fix.Iters,...
    out_ngmres_fix.time, round(out_ngmres_fix.FuncEvals), ...
    out_ngmres_fix.F, norm(out_ngmres_fix.G)/n);

%--------------------------------------
% Call NCG for comparison
%--------------------------------------
fprintf('+++ Start N-CG +++\n');
pncgtime = tic;
out_ncg = ncg_prec(fg,x0,params.ncg);
out_ncg.time = toc(pncgtime);
for i = 2:size(out_ncg.TraceFuncEvals,2)
    out_ncg.TraceFuncEvals(i) = out_ncg.TraceFuncEvals(i)...
        +out_ncg.TraceFuncEvals(i-1);
end
fprintf(' Iter   Time   FuncEvals       F(X)          ||G(X)||/N        \n');
fprintf('------  -----  --------- ---------------- ----------------\n');
fprintf('%6d %5.2g %9d %16.8g %16.8g\n', out_ncg.Iters,...
    out_ncg.time, round(out_ncg.FuncEvals), ...
    out_ncg.F, norm(out_ncg.G)/n);

%-----------------------------------------
% some figure output
%--------------------------------------
if params.shared.TraceGradNorm
    figure(figStart+1)
    set(gca, 'YScale', 'log');  
    semilogy(out_nesterov.TraceGradNorm(2:end),'-x')
    title('convergence of the norm of the gradient')
    hold on
    semilogy(out_ngmres_ls.TraceGradNorm(2:end),'-+')
    if params.ngmres.TraceRestarts
        loggMod=out_ngmres_ls.TraceGradNorm(2:end);
        loggMod(out_ngmres_ls.TraceRestarts==0)=NaN;
        semilogy(loggMod,'+b')
    end

    semilogy(out_ngmres_fix.TraceGradNorm(2:end),'-o')
    if params.ngmres.TraceRestarts
        loggMod=out_ngmres_fix.TraceGradNorm(2:end);
        loggMod(out_ngmres_fix.TraceRestarts==0)=NaN;
        semilogy(loggMod,'ob')
    end
    
    semilogy(out_ncg.TraceGradNorm(2:end),'-*')
    semilogy(out_lbfgs.TraceGradNorm(2:end),'-s')
    
    if params.ngmres.TraceRestarts
        legend('Nesterov - LS','NGMRES - LS','NGMRES - LS - restarts','NGMRES - Fixed','NGMRES - Fixed - restarts','NCG','LBGFS')
    else
        legend('Nesterov - LS','NGMRES - LS','NGMRES - Fixed','NCG','LBGFS')
    end
    xlabel('iterations')
    hold off

    figure(figStart+2)
    set(gca, 'YScale', 'log');    
    semilogy(out_nesterov.TraceFuncEvals(2:end),out_nesterov.TraceGradNorm(2:end),'-x')
    title('convergence of the norm of the gradient')
    hold on
    semilogy(out_ngmres_ls.TraceFuncEvals(2:end),out_ngmres_ls.TraceGradNorm(2:end),'-+')
    if params.ngmres.TraceRestarts
        loggMod=out_ngmres_ls.TraceGradNorm(2:end);
        loggMod(out_ngmres_ls.TraceRestarts==0)=NaN;
        semilogy(out_ngmres_ls.TraceFuncEvals(2:end),loggMod,'+b')
    end

    semilogy(out_ngmres_fix.TraceFuncEvals(2:end),out_ngmres_fix.TraceGradNorm(2:end),'-o')
    if params.ngmres.TraceRestarts
        loggMod=out_ngmres_fix.TraceGradNorm(2:end);
        loggMod(out_ngmres_fix.TraceRestarts==0)=NaN;
        semilogy(out_ngmres_fix.TraceFuncEvals(2:end),loggMod,'ob')
    end
    
    semilogy(out_ncg.TraceFuncEvals(2:end),out_ncg.TraceGradNorm(2:end),'-*')
    semilogy(out_lbfgs.TraceFuncEvals(2:end),out_lbfgs.TraceGradNorm(2:end),'-s')
    
    if params.ngmres.TraceRestarts
        legend('Nesterov - LS','NGMRES - LS','NGMRES - LS - restarts','NGMRES - Fixed','NGMRES - Fixed - restarts','NCG','LBGFS')
    else
        legend('Nesterov - LS','NGMRES - LS','NGMRES - Fixed','NCG','LBGFS')
    end
    xlabel('function evaluations')
    hold off

    % APPROXIMATE time plot (approximate because we assume the time per
    % itereation is constant)
    figure(figStart+3) 
    set(gca, 'YScale', 'log');
    nits=size(out_nesterov.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_nesterov.time,nits),out_nesterov.TraceGradNorm(2:end),'-x')
    title('convergence of the norm of the gradient')
    hold on
    nits=size(out_ngmres_ls.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_ngmres_ls.time,nits),out_ngmres_ls.TraceGradNorm(2:end),'-+')
    if params.ngmres.TraceRestarts
        loggMod=out_ngmres_ls.TraceGradNorm(2:end);
        loggMod(out_ngmres_ls.TraceRestarts==0)=NaN;
        semilogy(linspace(0,out_ngmres_ls.time,nits),loggMod,'+b')
    end

    nits=size(out_ngmres_fix.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_ngmres_fix.time,nits),out_ngmres_fix.TraceGradNorm(2:end),'-o')
    if params.ngmres.TraceRestarts
        loggMod=out_ngmres_fix.TraceGradNorm(2:end);
        loggMod(out_ngmres_fix.TraceRestarts==0)=NaN;
        semilogy(linspace(0,out_ngmres_fix.time,nits),loggMod,'ob')
    end
    
    nits=size(out_ncg.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_ncg.time,nits),out_ncg.TraceGradNorm(2:end),'-*')
    nits=size(out_lbfgs.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_lbfgs.time,nits),out_lbfgs.TraceGradNorm(2:end),'-s')
    
    if params.ngmres.TraceRestarts
        legend('Nesterov - LS','NGMRES - LS','NGMRES - LS - restarts','NGMRES - Fixed','NGMRES - Fixed - restarts','NCG','LBGFS')
    else
        legend('Nesterov - LS','NGMRES - LS','NGMRES - Fixed','NCG','LBGFS')
    end
    xlabel('time (s) (assuming constant time per iteration)')
    hold off
    
end

if params.shared.TraceFunc
    figure(figStart+4)
    set(gca, 'YScale', 'log');
    minval=min(min(out_ngmres_ls.TraceFunc),min(out_nesterov.TraceFunc(2:end)));
    semilogy(out_nesterov.TraceFunc(2:end)-minval,'-x')
    title('convergence towards the minimum value of f')
    hold on
    semilogy(out_ngmres_ls.TraceFunc(2:end)-min(out_ngmres_ls.TraceFunc(2:end)),'-+')
    if params.ngmres.TraceRestarts
        logfMod=out_ngmres_ls.TraceFunc(2:end)-min(out_ngmres_ls.TraceFunc(2:end));
        logfMod(out_ngmres_ls.TraceRestarts==0)=NaN;
        semilogy(logfMod,'+b')
    end

    semilogy(out_ngmres_fix.TraceFunc(2:end)-min(out_ngmres_fix.TraceFunc(2:end)),'-o')
    if params.ngmres.TraceRestarts
        logfMod=out_ngmres_fix.TraceFunc(2:end)-min(out_ngmres_fix.TraceFunc(2:end));
        logfMod(out_ngmres_fix.TraceRestarts==0)=NaN;
        semilogy(logfMod,'ob')
    end

    minval=min(min(out_ngmres_ls.TraceFunc),min(out_ncg.TraceFunc(2:end)));
    semilogy(out_ncg.TraceFunc(2:end)-minval,'-*')
    
    minval=min(min(out_ngmres_ls.TraceFunc),min(out_lbfgs.TraceFunc(2:end)));
    semilogy(out_lbfgs.TraceFunc(2:end)-minval,'-s')
    
    if params.ngmres.TraceRestarts
        legend('Nesterov - LS','NGMRES - LS','NGMRES - LS - restarts','NGMRES - Fixed','NGMRES - Fixed - restarts','NCG','LBGFS')
    else
        legend('Nesterov - LS','NGMRES - LS','NGMRES - Fixed','NCG','LBGFS')
    end
    xlabel('iterations')
    hold off

    figure(figStart+5)
    set(gca, 'YScale', 'log');
    minval=min(min(out_ngmres_ls.TraceFunc),min(out_nesterov.TraceFunc(2:end)));
    semilogy(out_nesterov.TraceFuncEvals(2:end),out_nesterov.TraceFunc(2:end)-minval,'-x')
    title('convergence towards the minimum value of f')
    hold on
    semilogy(out_ngmres_ls.TraceFuncEvals(2:end),out_ngmres_ls.TraceFunc(2:end)-min(out_ngmres_ls.TraceFunc(2:end)),'-+')
    if params.ngmres.TraceRestarts
        logfMod=out_ngmres_ls.TraceFunc(2:end)-min(out_ngmres_ls.TraceFunc(2:end));
        logfMod(out_ngmres_ls.TraceRestarts==0)=NaN;
        semilogy(out_ngmres_ls.TraceFuncEvals(2:end),logfMod,'+b')
    end

    semilogy(out_ngmres_fix.TraceFuncEvals(2:end),out_ngmres_fix.TraceFunc(2:end)-min(out_ngmres_fix.TraceFunc(2:end)),'-o')
    if params.ngmres.TraceRestarts
        logfMod=out_ngmres_fix.TraceFunc(2:end)-min(out_ngmres_fix.TraceFunc(2:end));
        logfMod(out_ngmres_fix.TraceRestarts==0)=NaN;
        semilogy(out_ngmres_fix.TraceFuncEvals(2:end),logfMod,'ob')
    end

    minval=min(min(out_ngmres_ls.TraceFunc),min(out_ncg.TraceFunc(2:end)));
    semilogy(out_ncg.TraceFuncEvals(2:end),out_ncg.TraceFunc(2:end)-minval,'-*')
    
    minval=min(min(out_ngmres_ls.TraceFunc),min(out_lbfgs.TraceFunc(2:end)));
    semilogy(out_lbfgs.TraceFuncEvals(2:end),out_lbfgs.TraceFunc(2:end)-minval,'-s')
    
    if params.ngmres.TraceRestarts
        legend('Nesterov - LS','NGMRES - LS','NGMRES - LS - restarts','NGMRES - Fixed','NGMRES - Fixed - restarts','NCG','LBGFS')
    else
        legend('Nesterov - LS','NGMRES - LS','NGMRES - Fixed','NCG','LBGFS')
    end
    xlabel('function evaluations')
    hold off

    % APPROXIMATE time plot (approximate because we assume the time per
    % itereation is constant)
    figure(figStart+6)
    set(gca, 'YScale', 'log');    
    minval=min(min(out_ngmres_ls.TraceFunc),min(out_nesterov.TraceFunc(2:end)));
    nits=size(out_nesterov.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_nesterov.time,nits),out_nesterov.TraceFunc(2:end)-minval,'-x')
    title('convergence towards the minimum value of f')
    hold on
    nits=size(out_ngmres_ls.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_ngmres_ls.time,nits),out_ngmres_ls.TraceFunc(2:end)-min(out_ngmres_ls.TraceFunc(2:end)),'-+')
    if params.ngmres.TraceRestarts
        logfMod=out_ngmres_ls.TraceFunc(2:end)-min(out_ngmres_ls.TraceFunc(2:end));
        logfMod(out_ngmres_ls.TraceRestarts==0)=NaN;
        semilogy(linspace(0,out_ngmres_ls.time,nits),logfMod,'+b')
    end

    nits=size(out_ngmres_fix.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_ngmres_fix.time,nits),out_ngmres_fix.TraceFunc(2:end)-min(out_ngmres_fix.TraceFunc(2:end)),'-o')
    if params.ngmres.TraceRestarts
        logfMod=out_ngmres_fix.TraceFunc(2:end)-min(out_ngmres_fix.TraceFunc(2:end));
        logfMod(out_ngmres_fix.TraceRestarts==0)=NaN;
        semilogy(linspace(0,out_ngmres_fix.time,nits),logfMod,'ob')
    end

    minval=min(min(out_ngmres_ls.TraceFunc),min(out_ncg.TraceFunc(2:end)));
    nits=size(out_ncg.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_ncg.time,nits),out_ncg.TraceFunc(2:end)-minval,'-*')
    
    minval=min(min(out_ngmres_ls.TraceFunc),min(out_lbfgs.TraceFunc(2:end)));
    nits=size(out_lbfgs.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_lbfgs.time,nits),out_lbfgs.TraceFunc(2:end)-minval,'-s')
    
    if params.ngmres.TraceRestarts
        legend('Nesterov - LS','NGMRES - LS','NGMRES - LS - restarts','NGMRES - Fixed','NGMRES - Fixed - restarts','NCG','LBGFS')
    else
        legend('Nesterov - LS','NGMRES - LS','NGMRES - Fixed','NCG','LBGFS')
    end
    xlabel('time (s) (assuming constant time per iteration)')
    hold off

end

%end