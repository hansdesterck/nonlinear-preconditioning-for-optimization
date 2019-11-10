function out = nesterov_prec(FUN,x0,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NESTEROV: Nesterov optimization method (accelerated gradient descent)
%
% OUT = nesterov(FUN,x0,varargin) minimizes a function using the Nesterov
%   optimization method with a nonlinear preconditioner as proposed in
%      -Mitchell, De Sterck, Ye, "Nesterov Acceleration of Alternating
%      Least Squares for Canonical Tensor Decomposition", arXiv:1810.05846.
%       
% INPUT:
%  x0: Initial guess (column vector)
%  [f,g] = FUN(x): FUN is a function handle to a function that computes
%     function value f and gradient vector g at point x
%
%  See POBLANO_PARAMS for further details on standard parameters.
%  Additionally, NESTEROV uses
%
%  'UsePrecond' - use nonlinear preconditioner to compute new unaccelerated
%   iterate (instead of the default gradient step) {false}
%
%  'FUNPrecond' - nonlinear preconditioner function handle {@(x) x}
%  [gprecond,nfev] = FUNprecond(x): FUNprecond
%     is a function handle to a function that computes a step direction,
%     -gprecond, from current approximation x; FUNprecond is the preconditioner
%     of the NGMRES method; nfev is the number of f and g evaluations
%
%  'NoPrecondStep' - choose how to set step length in default gradient step
%   to compute new unaccelerated iterate (when nonlinear preconditioner is
%   not used, i.e., when UsePrecond is false) {ls}
%       'ls' - steepest descent with More-Theunte linesearch
%       'fixed' - steepest descent with fixed step length based on norm(g)
%
%  'alpha' - fixed step size for gradient descent when UsePrecond is false
%   and a fixed step is chosen for the default gradient descent preconditioner
%   {10^-3}
%
%  'step_type' - choose the step length for beta in the Nesterov step {2}
%      [1] - Fixed step length, beta = 1
%      [2] - Adaptive step length based on gradient ratio
%      [3] - Nesterov-based step length (Nesterov sequence)
%      [4] - Linesearch-based step length
%
%  'restart_type' - choose the restart criteria {1}
%      [0] - No restart
%      [1] - Function restart
%      [2] - Gradient restart
%      [3] - X-based restart
%
%  'eta' - choose how much slack to allow before a restart {1}
%
%  'delay' - choose what iterate to compare against for restarting
%   criteria {1}
%
% PARAMS = nesterov('defaults') returns a structure containing the
%  default parameters for the particular Poblano method.
%
% OUTPUT:
% See POBLANO_OUT for details of the output parameters.
%
% Drew Mitchell and Hans De Sterck, 2018 - 2019
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create parser
params = inputParser;

% Set Poblano parameters
params = poblano_params(params);

% Set parameters for this method
params.addParamValue('UsePrecond',false,@islogical);
params.addParamValue('FUNPrecond',@(x) x, @(x) isa(x, 'function_handle'));
params.addParamValue('NoPrecondStep','ls',@(x) ismember(x,{'ls','fixed'}));
params.addParamValue('alpha',1e-3,@(x) x > 0);
params.addParamValue('step_type',2,@(x) any(x == [1,2,3,4]));
params.addParamValue('restart_type',1,@(x) any(x == [0,1,2,3]));
params.addParamValue('eta',1,@(x) x > 0);
params.addParamValue('delay',1,@(x) x > 0);

% Parse input
params.parse(varargin{:});

%% Check input arguments
if (nargin == 1) && isequal(FUN,'defaults') && (nargout == 1)
    out = params.Results;
    return;
elseif (nargin < 2)
    error('Error: invalid input arguments');
end

% Initialization
eta_upper = params.Results.eta + 0.1;
eta_lower = params.Results.eta;
eta = params.Results.eta;
nX = length(x0); % Number of components in the x vectors
alpha = params.Results.alpha;

Restarts = zeros(params.Results.MaxIters,1);

xk = x0;
[fk,gk_old] = feval(FUN,xk);

grad_mem = [];
func_mem = [];
memo = params.Results.delay;
grad_mem(1:memo) = norm(gk_old);
func_mem(1:memo) = fk;

out = poblano_out(xk,fk,gk_old,1,params);

% initialize 2nd iterate for use in gradient ratio and direction
x_old = xk;

if (params.Results.UsePrecond)
    %setup of x2
    [gk,nfev_precond] = feval(params.Results.FUNPrecond,xk);
    xk = xk - gk;
    [fk, gk] = feval(FUN,xk);
    out = poblano_out(xk,fk,gk,nfev_precond+1,params,out);
else
    [xk, fk, gk, nfev] = grad_precond(FUN, params, gk_old, fk, xk, alpha);
    out = poblano_out(xk,fk,gk,nfev,params,out);
end

% store gradient and function values in memory for use in delay comparison
grad_mem = [norm(gk) grad_mem];
grad_mem = grad_mem(1:memo);
func_mem = [fk func_mem];
func_mem = func_mem(1:memo);

% initialise Nesterov step parameters
if params.Results.step_type == 3
    lam_old = 0;
    lam_new=(1+sqrt(1+4*lam_old^2))/2;
end

% Main loop iteration
while out.ExitFlag == -1
    if strcmp(params.Results.Display, 'iter')
        disp('*** Restart ***')
    end
    
    % determine the step length based on user input
    if params.Results.step_type == 1
        beta = 1;
    elseif params.Results.step_type == 2
        beta = norm(gk)/norm(gk_old);
    elseif params.Results.step_type == 3
        beta = (lam_old-1)/lam_new;
        lam_old = lam_new;
        lam_new=(1+sqrt(1+4*lam_old^2))/2;
    end
    
    % update parameters
    gk_old = gk;
    fk_old = fk;
    d = xk - x_old;
    x_older = x_old;
    x_old = xk;
    % line search or alternative step
    if params.Results.step_type ~= 4
        xk=xk+beta*d;
        nfev_ls = 0;
    else
        pk = d;
        [xk,~,~,~,~,nfev_ls] = poblano_linesearch(FUN,xk,fk,gk,1,pk,params.Results);
    end
    
    % apply either the preconditioner or the gradient descent step.
    if (params.Results.UsePrecond)
        [gk,nfev_precond] = feval(params.Results.FUNPrecond,xk);
        xk = xk - gk;
        [fk, gk] = feval(FUN,xk);
        nfev = 1;
    else
        [fk,gk] = feval(FUN,xk);
        nfev = 1;
        [xk, fk, gk, nfev_precond] = grad_precond(FUN, params, gk, fk, xk, alpha);
    end
    
    % check restart criteria
    restart = restart_check(params.Results.restart_type,params.Results.eta,norm(gk),fk,xk,x_old,x_older,grad_mem(end),func_mem(end));
    if restart
        % throw away new iterate and apply preconditioner to previous
        % iterate
        xk = x_old;
        fk = fk_old;
        gk = gk_old;
        
        if (params.Results.UsePrecond)
            [gk,nfev_precond2] = feval(params.Results.FUNPrecond,xk);
            xk = xk - gk;
            [fk, gk] = feval(FUN,xk);
            nfev = nfev + 1;
            nfev_precond = nfev_precond + nfev_precond2;
        else
            [xk, fk, gk, nfev_precond2] = grad_precond(FUN, params, gk, fk, xk, alpha);
            nfev_precond = nfev_precond + nfev_precond2;
        end
        
        % reset the nesterov step sequence
        if params.Results.step_type == 3
            lam_old = 0;
            lam_new=(1+sqrt(1+4*lam_old^2))/2;
        end
        
        % eta slack update
        if eta_lower ~= 1
            eta = eta_upper;
        end
    else
    end
    
    if eta > eta_lower && eta_lower ~= 1
        eta = eta - 0.02;
    end
    % update delay memory
    grad_mem = [norm(gk) grad_mem];
    grad_mem = grad_mem(1:memo);
    func_mem = [fk func_mem];
    func_mem = func_mem(1:memo);
    
    % Provide some output and get some log information
    out = poblano_out(xk,fk,gk,nfev_precond + nfev+nfev_ls,params,out);
end

function [xk, fk, gk, nfev] = grad_precond(FUN, params, gk, fk, xk,alpha)
% gradient preconditioner for the Nesterov method.
if strcmp(params.Results.NoPrecondStep, 'ls') % use line search method
    pk = -gk;
    [xk,fk,gk,~,~,nfev] = poblano_linesearch(FUN,xk,fk,gk,1,pk,params.Results);
elseif strcmp(params.Results.NoPrecondStep, 'fixed') % use fixed step method
    xk = xk - alpha*gk;
    nfev=1;
    [fk,gk] = feval(FUN,xk);
end

function restart = restart_check(restart_type,eta,gk_new,fk,xk,x_old,x_older,grad_mem,func_mem)
%restart type 0: No Restart
%restart type 1: Function Restart
%restart type 2: Gradient Restart
%restart type 3: X-based Restart
restart = 0;
if restart_type == 1
    if fk >= func_mem *eta
        restart = 1;
    end
elseif restart_type == 2
    if gk_new >= grad_mem *eta
        restart = 1;
    end
elseif restart_type == 3
    if norm(xk - x_old) < norm(x_old - x_older)*eta
        restart = 1;
    end
end
