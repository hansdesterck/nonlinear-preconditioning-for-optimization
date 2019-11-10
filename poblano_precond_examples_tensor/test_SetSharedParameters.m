function params = test_SetSharedParameters(params)

%NGMRES parameters
params.ngmres.MaxIters = params.shared.MaxIters;
params.ngmres.MaxFuncEvals = params.shared.MaxFuncEvals;
params.ngmres.TraceFuncEvals = params.shared.TraceFuncEvals;
params.ngmres.TraceFunc = params.shared.TraceFunc;
params.ngmres.TraceGradNorm = params.shared.TraceGradNorm;
params.ngmres.RelFuncTol = params.shared.RelFuncTol;
params.ngmres.StopTol = params.shared.GNormTol;
params.ngmres.LineSearch_ftol = params.ls.LineSearch_ftol;
params.ngmres.LineSearch_gtol = params.ls.LineSearch_gtol;
params.ngmres.LineSearch_initialstep = params.ls.LineSearch_initialstep;
params.ngmres.LineSearch_maxfev = params.ls.LineSearch_maxfev;
params.ngmres.LineSearch_method = params.ls.LineSearch_method;
params.ngmres.LineSearch_stpmax = params.ls.LineSearch_stpmax;
params.ngmres.LineSearch_stpmin = params.ls.LineSearch_stpmin;
params.ngmres.LineSearch_xtol = params.ls.LineSearch_xtol;

%NCG parameters
params.ncg.MaxIters = params.shared.MaxIters;
params.ncg.MaxFuncEvals = params.shared.MaxFuncEvals;
params.ncg.TraceFuncEvals = params.shared.TraceFuncEvals;
params.ncg.TraceFunc = params.shared.TraceFunc;
params.ncg.TraceGradNorm = params.shared.TraceGradNorm;
params.ncg.RelFuncTol = params.shared.RelFuncTol;
params.ncg.StopTol = params.shared.GNormTol;
params.ncg.LineSearch_ftol = params.ls.LineSearch_ftol;
params.ncg.LineSearch_gtol = params.ls.LineSearch_gtol;
params.ncg.LineSearch_initialstep = params.ls.LineSearch_initialstep;
params.ncg.LineSearch_maxfev = params.ls.LineSearch_maxfev;
params.ncg.LineSearch_method = params.ls.LineSearch_method;
params.ncg.LineSearch_stpmax = params.ls.LineSearch_stpmax;
params.ncg.LineSearch_stpmin = params.ls.LineSearch_stpmin;
params.ncg.LineSearch_xtol = params.ls.LineSearch_xtol;

%LBFGS parameters
params.lbfgs.MaxIters = params.shared.MaxIters;
params.lbfgs.MaxFuncEvals = params.shared.MaxFuncEvals;
params.lbfgs.TraceFuncEvals = params.shared.TraceFuncEvals;
params.lbfgs.TraceFunc = params.shared.TraceFunc;
params.lbfgs.TraceRelFunc = params.shared.TraceRelFunc;
params.lbfgs.TraceGradNorm = params.shared.TraceGradNorm;
params.lbfgs.RelFuncTol = params.shared.RelFuncTol;
params.lbfgs.StopTol = params.shared.GNormTol;
params.lbfgs.LineSearch_ftol = params.ls.LineSearch_ftol;
params.lbfgs.LineSearch_gtol = params.ls.LineSearch_gtol;
params.lbfgs.LineSearch_initialstep = params.ls.LineSearch_initialstep;
params.lbfgs.LineSearch_maxfev = params.ls.LineSearch_maxfev;
params.lbfgs.LineSearch_method = params.ls.LineSearch_method;
params.lbfgs.LineSearch_stpmax = params.ls.LineSearch_stpmax;
params.lbfgs.LineSearch_stpmin = params.ls.LineSearch_stpmin;
params.lbfgs.LineSearch_xtol = params.ls.LineSearch_xtol;


%Nesterov parameters
params.nesterov.MaxIters = params.shared.MaxIters;
params.nesterov.MaxFuncEvals = params.shared.MaxFuncEvals;
params.nesterov.TraceFuncEvals = params.shared.TraceFuncEvals;
params.nesterov.TraceFunc = params.shared.TraceFunc;
params.nesterov.TraceRelFunc = params.shared.TraceRelFunc;
params.nesterov.TraceGradNorm = params.shared.TraceGradNorm;
params.nesterov.RelFuncTol = params.shared.RelFuncTol;
params.nesterov.StopTol = params.shared.GNormTol;
params.nesterov.LineSearch_ftol = params.ls.LineSearch_ftol;
params.nesterov.LineSearch_gtol = params.ls.LineSearch_gtol;
params.nesterov.LineSearch_initialstep = params.ls.LineSearch_initialstep;
params.nesterov.LineSearch_maxfev = params.ls.LineSearch_maxfev;
params.nesterov.LineSearch_method = params.ls.LineSearch_method;
params.nesterov.LineSearch_stpmax = params.ls.LineSearch_stpmax;
params.nesterov.LineSearch_stpmin = params.ls.LineSearch_stpmin;
params.nesterov.LineSearch_xtol = params.ls.LineSearch_xtol;
