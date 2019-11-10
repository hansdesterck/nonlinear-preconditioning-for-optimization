%function test_driver_CPtensor
clc
clear all
addpath ../poblano_toolbox
addpath ../poblano_toolbox_ext
addpath ../tensor_toolbox-3.1
figStart=0;
%------------------------------
% Parameters
%------------------------------
%Shared parameters
params.shared.MaxIters = 200;
params.shared.MaxFuncEvals = 4e3;
params.shared.RelFuncTol = -1;
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
params.ngmres.UsePrecond = true;
params.ngmres.NoPrecondStep = 'ls'; %if above is false. Options: 'ls','fixed'
params.ngmres.NPrecond = 1;
params.ngmres.AlgorithmVariant = 'ngmres'; % Options: 'ngmres', 'anderson'

% NCG parameters (for comparing NCG with NGMRES)
params.ncg = ncg_prec('defaults');
params.ncg.RestartNW = false;
params.ncg.Display = 'off'; % Options: 'iter', 'final' or 'off'
params.ncg.UsePrecond = true;

% LBFGS parameters (for comparing LBFGS with NGMRES)
params.lbfgs = lbfgs_prec('defaults');
params.lbfgs.M = 5; % Window size
params.lbfgs.Display = 'off'; % Options: 'iter', 'final' or 'off'
params.lbfgs.LS_precond = 'modBT';

% Nesterov parameters
params.nesterov = nesterov_prec('defaults');
params.nesterov.Display = 'off';  % Options: 'iter', 'final' or 'off'
params.nesterov.UsePrecond = true;
params.nesterov.NoPrecondStep = 'ls'; %if above is false. Options: 'ls','fixed'
params.nesterov.eta = 1; % value of eta
params.nesterov.delay = 1; % 1: no delay
params.nesterov.step_type = 2; % 1: step 1; 2: gradient ratio; 3: Nesterov sequence; 4: line search on beta
params.nesterov.restart_type = 1; % 0: no restart; 1: function; 2: gradient; 3: speed (x-based)

%Load Shared Parameters;
params = test_SetSharedParameters(params);

%------------------------------
% Create a test tensor
% (This is the test from Tomasi and Bro (2006) and Acar et al. (2011) for
% a random dense tensor modified to get specified collinearity between the columns)
%------------------------------
params.I = 50;
params.collinearity = 0.9;
params.R = 3;
params.l1 = 1;
params.l2 = 1;
params.initSeed = 0;
rs = RandStream('mt19937ar','Seed',params.initSeed);
RandStream.setGlobalStream(rs);
tensorpars = [params.I params.collinearity params.R params.l1 params.l2];
[T, ~] = test_CreateTensor(tensorpars);
nT2 = norm(T)^2;

%--------------------------------------
% Generate the random initial guess
%--------------------------------------
nX = params.R*sum(size(T));
x0 = rand(nX,1);

% The fg function
fg = @(x) tt_cp_fun(x,T,nT2);

% Nonlinear preconditioner using ALS (use our custom implementation, or the
% tensor toolbox implementation (TTB))
precondopt = 'TTB'; % options are 'ours', 'TTB'
g_precond = @(x) test_CPtensor_preconditioner(x,T,params.R,1,precondopt);

params.lbfgs.FUNPrecond = g_precond;
params.ncg.FUNPrecond = g_precond;
params.ngmres.FUNPrecond = g_precond;
params.nesterov.FUNPrecond = g_precond;

%---------------------------------------------------------------------%
% More-Thuente Line Search Results
%---------------------------------------------------------------------%
fprintf('+++ CVSRCH TESTS +++\n\n');
params.ncg.LineSearch_method = 'more-thuente';
params.ngmres.LineSearch_method = 'more-thuente';

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
    out_nesterov.F, norm(out_nesterov.G)/nX);
%--------------------------------------
% Call the NGMRES method
%--------------------------------------
fprintf('+++ Start N-GMRES +++\n');
ngmrestime = tic;
out_ngmres = ngmres_prec(fg, x0, params.ngmres);
out_ngmres.time = toc(ngmrestime);
for i = 2:size(out_ngmres.TraceFuncEvals,2)
    out_ngmres.TraceFuncEvals(i) = out_ngmres.TraceFuncEvals(i)+out_ngmres.TraceFuncEvals(i-1);
end
fprintf(' Iter   Time   FuncEvals       F(X)          ||G(X)||/N        \n');
fprintf('------  -----  --------- ---------------- ----------------\n');
fprintf('%6d %5.2g %9d %16.8g %16.8g\n', out_ngmres.Iters,...
    out_ngmres.time, round(out_ngmres.FuncEvals), ...
    out_ngmres.F, norm(out_ngmres.G)/nX);

%--------------------------------------
% Call the Anderson Acceleration method
%--------------------------------------
fprintf('+++ Start Anderson +++\n');
andersontime = tic;
params.ngmres.AlgorithmVariant = 'anderson'; % Options: 'ngmres', 'anderson'
out_anderson = ngmres_prec(fg, x0, params.ngmres);
out_anderson.time = toc(andersontime);
for i = 2:size(out_anderson.TraceFuncEvals,2)
    out_anderson.TraceFuncEvals(i) = out_anderson.TraceFuncEvals(i)+out_anderson.TraceFuncEvals(i-1);
end
fprintf(' Iter   Time   FuncEvals       F(X)          ||G(X)||/N        \n');
fprintf('------  -----  --------- ---------------- ----------------\n');
fprintf('%6d %5.2g %9d %16.8g %16.8g\n', out_anderson.Iters,...
    out_anderson.time, round(out_anderson.FuncEvals), ...
    out_anderson.F, norm(out_anderson.G)/nX);

%--------------------------------------
% Call the ALS method
%--------------------------------------
fprintf('+++ Start ALS +++\n');
x = x0;
for k=1:8*params.shared.MaxIters
    x = x - g_precond(x);
    [f,g] = fg(x);
    fALS(k) = f;
    gALS(k) = norm(g);
end

% ALS without fg calculation, for timing
x = x0;
alstime = tic;
for k=1:8*params.shared.MaxIters
    x = x - g_precond(x);
end
alstime = toc(alstime);

%--------------------------------------
% Call Preconditioned NCG for comparison
%--------------------------------------
updates = {'HS_LP','HS_TP'};
nupdates = length(updates);

for uv = 1:nupdates
    fprintf('+++ Start Preconditioned N-CG with %s Update +++\n',updates{uv});
    params.ncg.Update = updates{uv};
    pncgtime = tic;
    out_ncg{uv} = ncg_prec(fg,x0,params.ncg);
    out_ncg{uv}.time = toc(pncgtime);
    out_ncg{uv}.Update = updates{uv};
    for i = 2:size(out_ncg{uv}.TraceFuncEvals,2)
        out_ncg{uv}.TraceFuncEvals(i) = out_ncg{uv}.TraceFuncEvals(i)...
            +out_ncg{uv}.TraceFuncEvals(i-1);
    end
    fprintf(' Iter   Time   FuncEvals       F(X)          ||G(X)||/N        \n');
    fprintf('------  -----  --------- ---------------- ----------------\n');
    fprintf('%6d %5.2g %9d %16.8g %16.8g\n', out_ncg{uv}.Iters,...
        out_ncg{uv}.time, round(out_ncg{uv}.FuncEvals), ...
        out_ncg{uv}.F, norm(out_ncg{uv}.G)/nX);
end

%--------------------------------------
% Call LBFGS
%--------------------------------------
fprintf('+++ Start LP-LBFGS +++\n');
lbfgstime = tic;
params.lbfgs.PrecondType = 'LP';
out_lbfgs_LP = lbfgs_prec(fg,x0,params.lbfgs);
out_lbfgs_LP.time = toc(lbfgstime);
for i=2:size(out_lbfgs_LP.TraceFuncEvals,2)
    out_lbfgs_LP.TraceFuncEvals(i)= out_lbfgs_LP.TraceFuncEvals(i)+...
        out_lbfgs_LP.TraceFuncEvals(i-1);
end
fprintf(' Iter   Time   FuncEvals       F(X)          ||G(X)||/N        \n');
fprintf('------  -----  --------- ---------------- ----------------\n');
fprintf('%6d %5.2g %9d %16.8g %16.8g\n', out_lbfgs_LP.Iters,...
    out_lbfgs_LP.time, round(out_lbfgs_LP.FuncEvals), ...
    out_lbfgs_LP.F, norm(out_lbfgs_LP.G)/nX);


fprintf('+++ Start TP-LBFGS +++\n');
lbfgstime = tic;
params.lbfgs.PrecondType = 'TP';
out_lbfgs_TP = lbfgs_prec(fg,x0,params.lbfgs);
out_lbfgs_TP.time = toc(lbfgstime);
for i=2:size(out_lbfgs_TP.TraceFuncEvals,2)
    out_lbfgs_TP.TraceFuncEvals(i)= out_lbfgs_TP.TraceFuncEvals(i)+...
        out_lbfgs_TP.TraceFuncEvals(i-1);
end
fprintf(' Iter   Time   FuncEvals       F(X)          ||G(X)||/N        \n');
fprintf('------  -----  --------- ---------------- ----------------\n');
fprintf('%6d %5.2g %9d %16.8g %16.8g\n', out_lbfgs_TP.Iters, ...
    out_lbfgs_TP.time, round(out_lbfgs_TP.FuncEvals), ...
    out_lbfgs_TP.F, norm(out_lbfgs_TP.G)/nX);

%-----------------------------------------
% some figure output
%--------------------------------------
if params.shared.TraceGradNorm
    figure(figStart+1)
    set(gca, 'YScale', 'log');
    semilogy(out_nesterov.TraceGradNorm(2:end),'-x')
    hold on
    semilogy(out_ngmres.TraceGradNorm(2:end),'-+')
    title('convergence of the norm of the gradient')    
    if params.ngmres.TraceRestarts
        loggMod=out_ngmres.TraceGradNorm(2:end);
        loggMod(out_ngmres.TraceRestarts==0)=NaN;
        semilogy(loggMod,'+b')
    end
    
    semilogy(out_anderson.TraceGradNorm(2:end),'-+')
    semilogy(out_ncg{1}.TraceGradNorm(2:end),'-*')
    semilogy(out_ncg{2}.TraceGradNorm(2:end),'-o')
    semilogy(out_lbfgs_LP.TraceGradNorm(2:end),'-s')
    semilogy(out_lbfgs_TP.TraceGradNorm(2:end),'-d')
    semilogy(gALS(2:params.shared.MaxIters),'-')
    
    if params.ngmres.TraceRestarts
        legend('Nesterov-ALS','NGMRES-ALS','NGMRES - restarts','Anderson-ALS','LP-NCG-ALS','TP-NCG-ALS','LP-LBFGS-ALS','TP-LBFGS-ALS','ALS')
    else
        legend('Nesterov-ALS','NGMRES-ALS','Anderson-ALS','LP-NCG-ALS','TP-NCG-ALS','LP-LBFGS-ALS','TP-LBFGS-ALS','ALS')
    end
    xlabel('iterations')
    hold off

    figure(figStart+2)
    set(gca, 'YScale', 'log');
    semilogy(out_nesterov.TraceFuncEvals(2:end),out_nesterov.TraceGradNorm(2:end),'-x')
    hold on
    semilogy(out_ngmres.TraceFuncEvals(2:end),out_ngmres.TraceGradNorm(2:end),'-+')
    title('convergence of the norm of the gradient')    
    if params.ngmres.TraceRestarts
        loggMod=out_ngmres.TraceGradNorm(2:end);
        loggMod(out_ngmres.TraceRestarts==0)=NaN;
        semilogy(out_ngmres.TraceFuncEvals(2:end),loggMod,'+b')
    end
    
    semilogy(out_anderson.TraceFuncEvals(2:end),out_anderson.TraceGradNorm(2:end),'-+')
    semilogy(out_ncg{1}.TraceFuncEvals(2:end),out_ncg{1}.TraceGradNorm(2:end),'-*')
    semilogy(out_ncg{2}.TraceFuncEvals(2:end),out_ncg{2}.TraceGradNorm(2:end),'-o')
    semilogy(out_lbfgs_LP.TraceFuncEvals(2:end),out_lbfgs_LP.TraceGradNorm(2:end),'-s')
    semilogy(out_lbfgs_TP.TraceFuncEvals(2:end),out_lbfgs_TP.TraceGradNorm(2:end),'-d')
    semilogy(gALS(2:end),'-')
    
    if params.ngmres.TraceRestarts
        legend('Nesterov-ALS','NGMRES-ALS','NGMRES - restarts','Anderson-ALS','LP-NCG-ALS','TP-NCG-ALS','LP-LBFGS-ALS','TP-LBFGS-ALS','ALS')
    else
        legend('Nesterov-ALS','NGMRES-ALS','Anderson-ALS','LP-NCG-ALS','TP-NCG-ALS','LP-LBFGS-ALS','TP-LBFGS-ALS','ALS')
    end
    xlabel('function evaluations (assuming one ALS is equivalent to one fg)')
    hold off

    % APPROXIMATE time plot (approximate because we assume the time per
    % itereation is constant)
    figure(figStart+3)
    set(gca, 'YScale', 'log');
    nits=size(out_nesterov.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_nesterov.time,nits),out_nesterov.TraceGradNorm(2:end),'-x')
    hold on
    nits=size(out_ngmres.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_ngmres.time,nits),out_ngmres.TraceGradNorm(2:end),'-+')
    title('convergence of the norm of the gradient')    
    if params.ngmres.TraceRestarts
        loggMod=out_ngmres.TraceGradNorm(2:end);
        loggMod(out_ngmres.TraceRestarts==0)=NaN;
        semilogy(linspace(0,out_ngmres.time,nits),loggMod,'+b')
    end
    nits=size(out_anderson.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_anderson.time,nits),out_anderson.TraceGradNorm(2:end),'-+')

    nits=size(out_ncg{1}.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_ncg{1}.time,nits),out_ncg{1}.TraceGradNorm(2:end),'-*')
    
    nits=size(out_ncg{2}.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_ncg{2}.time,nits),out_ncg{2}.TraceGradNorm(2:end),'-o')
    
    nits=size(out_lbfgs_LP.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_lbfgs_LP.time,nits),out_lbfgs_LP.TraceGradNorm(2:end),'-s')
    
    nits=size(out_lbfgs_TP.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_lbfgs_TP.time,nits),out_lbfgs_TP.TraceGradNorm(2:end),'-d')
    
    nits=size(fALS(2:3*params.shared.MaxIters),2);
    semilogy(linspace(0,alstime*3/8,nits),gALS(2:3*params.shared.MaxIters),'-')
    
    if params.ngmres.TraceRestarts
        legend('Nesterov-ALS','NGMRES-ALS','NGMRES - restarts','Anderson-ALS','LP-NCG-ALS','TP-NCG-ALS','LP-LBFGS-ALS','TP-LBFGS-ALS','ALS')
    else
        legend('Nesterov-ALS','NGMRES-ALS','Anderson-ALS','LP-NCG-ALS','TP-NCG-ALS','LP-LBFGS-ALS','TP-LBFGS-ALS','ALS')
    end
    xlabel('time (s) (assuming constant time per iteration)')
    hold off

end

if params.shared.TraceFunc
    figure(figStart+4)
    set(gca, 'YScale', 'log');
    minval=min([min(out_ngmres.TraceFunc),min(out_anderson.TraceFunc),min(out_nesterov.TraceFunc(2:end)),...
        min(out_ncg{1}.TraceFunc(2:end)),min(out_ncg{2}.TraceFunc(2:end)),...
        min(out_lbfgs_LP.TraceFunc(2:end)),min(out_lbfgs_TP.TraceFunc(2:end)),...
        min(fALS)]);
    
    semilogy(out_nesterov.TraceFunc(2:end)-minval,'-x')
    hold on
    semilogy(out_ngmres.TraceFunc(2:end)-min(out_ngmres.TraceFunc(2:end)),'-+')  
    title('convergence towards the minimum value of f')    
    if params.ngmres.TraceRestarts
        logfMod=out_ngmres.TraceFunc(2:end)-min(out_ngmres.TraceFunc(2:end));
        logfMod(out_ngmres.TraceRestarts==0)=NaN;
        semilogy(logfMod,'+b')
    end
    semilogy(out_anderson.TraceFunc(2:end)-min(out_anderson.TraceFunc(2:end)),'-+')  
    semilogy(out_ncg{1}.TraceFunc(2:end)-minval,'-*')    
    semilogy(out_ncg{2}.TraceFunc(2:end)-minval,'-o')
    semilogy(out_lbfgs_LP.TraceFunc(2:end)-minval,'-s')    
    semilogy(out_lbfgs_TP.TraceFunc(2:end)-minval,'-d')    
    semilogy(fALS(2:params.shared.MaxIters)-minval,'-')
    
    if params.ngmres.TraceRestarts
        legend('Nesterov-ALS','NGMRES-ALS','NGMRES - restarts','Anderson-ALS','LP-NCG-ALS','TP-NCG-ALS','LP-LBFGS-ALS','TP-LBFGS-ALS','ALS')
    else
        legend('Nesterov-ALS','NGMRES-ALS','Anderson-ALS','LP-NCG-ALS','TP-NCG-ALS','LP-LBFGS-ALS','TP-LBFGS-ALS','ALS')
    end
    xlabel('iterations')
    hold off

    figure(figStart+5)
    set(gca, 'YScale', 'log');
    semilogy(out_nesterov.TraceFuncEvals(2:end),out_nesterov.TraceFunc(2:end)-minval,'-x')
    hold on
    semilogy(out_ngmres.TraceFuncEvals(2:end),out_ngmres.TraceFunc(2:end)-min(out_ngmres.TraceFunc(2:end)),'-+')
    title('convergence towards the minimum value of f')    
    if params.ngmres.TraceRestarts
        logfMod=out_ngmres.TraceFunc(2:end)-min(out_ngmres.TraceFunc(2:end));
        logfMod(out_ngmres.TraceRestarts==0)=NaN;
        semilogy(out_ngmres.TraceFuncEvals(2:end),logfMod,'+b')
    end
    semilogy(out_anderson.TraceFuncEvals(2:end),out_anderson.TraceFunc(2:end)-min(out_anderson.TraceFunc(2:end)),'-+')
    semilogy(out_ncg{1}.TraceFuncEvals(2:end),out_ncg{1}.TraceFunc(2:end)-minval,'-*')
    semilogy(out_ncg{2}.TraceFuncEvals(2:end),out_ncg{2}.TraceFunc(2:end)-minval,'-o')
    semilogy(out_lbfgs_LP.TraceFuncEvals(2:end),out_lbfgs_LP.TraceFunc(2:end)-minval,'-s')
    semilogy(out_lbfgs_TP.TraceFuncEvals(2:end),out_lbfgs_TP.TraceFunc(2:end)-minval,'-d')
    semilogy(fALS(2:end)-minval,'-')
    
    if params.ngmres.TraceRestarts
        legend('Nesterov-ALS','NGMRES-ALS','NGMRES - restarts','Anderson-ALS','LP-NCG-ALS','TP-NCG-ALS','LP-LBFGS-ALS','TP-LBFGS-ALS','ALS')
    else
        legend('Nesterov-ALS','NGMRES-ALS','Anderson-ALS','LP-NCG-ALS','TP-NCG-ALS','LP-LBFGS-ALS','TP-LBFGS-ALS','ALS')
    end
    xlabel('function evaluations (assuming one ALS is equivalent to one fg)')
    hold off

    % APPROXIMATE time plot (approximate because we assume the time per
    % itereation is constant)
    figure(figStart+6)
    set(gca, 'YScale', 'log');
    nits=size(out_nesterov.TraceGradNorm(2:end),2);
    semilogy(linspace(0,out_nesterov.time,nits),out_nesterov.TraceFunc(2:end)-minval,'-x')
    hold on
    nits=size(out_ngmres.TraceFunc(2:end),2);
    semilogy(linspace(0,out_ngmres.time,nits),out_ngmres.TraceFunc(2:end)-min(out_ngmres.TraceFunc(2:end)),'-+')
    title('convergence towards the minimum value of f')    
    if params.ngmres.TraceRestarts
        logfMod=out_ngmres.TraceFunc(2:end)-min(out_ngmres.TraceFunc(2:end));
        logfMod(out_ngmres.TraceRestarts==0)=NaN;
        semilogy(linspace(0,out_ngmres.time,nits),logfMod,'+b')
    end
    nits=size(out_anderson.TraceFunc(2:end),2);
    semilogy(linspace(0,out_anderson.time,nits),out_anderson.TraceFunc(2:end)-min(out_anderson.TraceFunc(2:end)),'-+')
    nits=size(out_ncg{1}.TraceFunc(2:end),2);
    semilogy(linspace(0,out_ncg{1}.time,nits),out_ncg{1}.TraceFunc(2:end)-minval,'-*')    
    nits=size(out_ncg{2}.TraceFunc(2:end),2);
    semilogy(linspace(0,out_ncg{2}.time,nits),out_ncg{2}.TraceFunc(2:end)-minval,'-o')
    nits=size(out_lbfgs_LP.TraceFunc(2:end),2);
    semilogy(linspace(0,out_lbfgs_LP.time,nits),out_lbfgs_LP.TraceFunc(2:end)-minval,'-s')
    nits=size(out_lbfgs_TP.TraceFunc(2:end),2);
    semilogy(linspace(0,out_lbfgs_TP.time,nits),out_lbfgs_TP.TraceFunc(2:end)-minval,'-d')
    nits=size(fALS(2:3*params.shared.MaxIters),2);
    semilogy(linspace(0,alstime*3/8,nits),fALS(2:3*params.shared.MaxIters)-minval,'-')
    
    if params.ngmres.TraceRestarts
        legend('Nesterov-ALS','NGMRES-ALS','NGMRES - restarts','Anderson-ALS','LP-NCG-ALS','TP-NCG-ALS','LP-LBFGS-ALS','TP-LBFGS-ALS','ALS')
    else
        legend('Nesterov-ALS','NGMRES-ALS','Anderson-ALS','LP-NCG-ALS','TP-NCG-ALS','LP-LBFGS-ALS','TP-LBFGS-ALS','ALS')
    end
    xlabel('time (s) (assuming constant time per iteration)')
    hold off

end
