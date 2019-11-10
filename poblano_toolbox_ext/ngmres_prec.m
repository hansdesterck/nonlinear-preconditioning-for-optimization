function out = ngmres_prec(FUN,x0,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NGMRES: Nonlinear GMRES optimization method
%
% OUT = NGMRES(FUN,x0,varargin) minimizes a function using the nonlinear GMRES
%   (N-GMRES) optimization method proposed and analyzed in
%      -Hans De Sterck, "A Nonlinear GMRES Optimization Algorithm for Canonical
%       Tensor Decomposition", SIAM J. Sci. Comput. 34, A1351-A1379, 2012.
%      -Hans De Sterck, "Steepest Descent Preconditioning for Nonlinear GMRES
%       Optimization", Numerical Linear Algebra with Applications 20, 453-471, 2013.
%
% INPUT:
%  x0: Initial guess (column vector)
%  [f,g] = FUN(x): FUN is a function handle to a function that computes
%     function value f and gradient vector g at point x
%
%  See POBLANO_PARAMS for further details on standard parameters.
%  Additionally, NGMRES uses
%
%  'UsePrecond' - use nonlinear preconditioner to compute new unaccelerated
%   iterate (instead of the default gradient step) {false}
%
%  'FUNPrecond' - nonlinear preconditioner function handle {@(x) x}
%  [gprecond,nfev] = FUNprecond(x): FUNprecond
%     is a function handle to a function that computes a step direction
%     -gprecond from current approximation x; FUNprecond is the preconditioner
%     of the NGMRES method; nfev is the number of f and g evaluations
%
%  'NoPrecondStep' - choose how to set step length in default gradient step
%   to compute new unaccelerated iterate (when nonlinear preconditioner is
%   not used, i.e., when UsePrecond is false) {ls}
%       'ls' - steepest descent with More-Theunte linesearch
%       'fixed' - steepest descent with fixed step length based on norm(g)
%
%  'W' - Maximum window size {20}.
%
%  'TraceRestarts' - Flag to include the history of the restarts {'false'}
%
%  'NPrecond' - number of preconditioner steps per NGMRES iteration (can be used
%   with default gradient steps or nonlinear preconditioner steps) {1}
%
%  'AlgorithmVariant' - either 'ngmres' (default) or 'anderson' (for Anderson acceleration) 
%
% PARAMS = NGMRES('defaults') returns a structure containing the
%  default parameters for the particular Poblano method.
%
% OUTPUT:
% See POBLANO_OUT for details of the output parameters.
%
% Hans De Sterck and Alexander Howse, 2011-2019
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
params.addParamValue('W',20,@(x) x > 0);
params.addParamValue('TraceRestarts',false,@islogical);
params.addParamValue('NPrecond',1,@(x) x > 0);
params.addParamValue('AlgorithmVariant','ngmres',@(x) ismember(x,{'ngmres','anderson'}));

% Parse input
params.parse(varargin{:});

%% Check input arguments
if (nargin == 1) && isequal(FUN,'defaults') && (nargout == 1)
    out = params.Results;
    return;
elseif (nargin < 2)
    error('Error: invalid input arguments');
end

% Fixed parameters used in the algorithm
epsi = 1e-12; % Parameter used in regularizing the normal equation system

% Initialization
nX = length(x0); % Number of components in the x vectors
Xwindow = zeros(nX,params.Results.W); %Solution vectors in the window
Gwindow = zeros(nX,params.Results.W); % Gradient vectors in the window
Restarts = zeros(params.Results.MaxIters,1);

xk = x0;
[fk,gk] = feval(FUN,xk);
out = poblano_out(xk,fk,gk,1,params);

while out.ExitFlag == -1
    % Start a new window
    Restarts(out.Iters+1)=1;
    if strcmp(params.Results.Display, 'iter')
        disp('*** Restart ***')
    end
    
    Xwindow(:,1) = xk; % The current iterate is the first vector in the new window
    Gwindow(:,1) = gk; % Also store its gradient
    GtGwind(1,1) = gk'*gk;
    
    restart = false; % No restart for now
    curW = 1; % Current window size (Starts at 1, grows up to params.Results.W)
    
    % Start a sequence of accelerated updates, build towards window size W, but exit if a restart is required
    cntLoc = 0; % Counter of number of iterations since the last restart
    while ~restart && out.ExitFlag == -1
        cntLoc = cntLoc+1;
        
        % STEP I: Get a new unaccelerated iterate
        %-------------------------------------------------
        % Compute its function value and gradient vector
        
        nfev_precond = 0;
        for i = 1:params.Results.NPrecond
            xkold = xk;
            fkold = fk;
            gkold = gk;
            if (~params.Results.UsePrecond)
                if strcmp(params.Results.NoPrecondStep, 'ls')
                    pk = -gk;
                    [xk,fk,gk,~,~,nfev] = poblano_linesearch(FUN,xk,fk,gk,1,pk,params.Results);
                else % use fixed step method
                    precStep1=1e-4; % step in steepest descent preconditioner without line search
                    precStep2=1e0;  % step factor in steepest descent preconditioner without line search
                    ng = norm(gk);
                    xk = xk - min(precStep1,precStep2*ng)*gk/ng;
                    nfev=1;
                    [fk,gk] = feval(FUN,xk);
                end
            else
                [gprecond,nfev] = feval(params.Results.FUNPrecond,xk);
                xk = xk - gprecond;
                [fk,gk] = feval(FUN,xk);
                nfev = nfev+1;
            end
            nfev_precond  = nfev_precond+nfev;
        end
        % for Anderson Acceleration:
        xk_prec = xk;
        gk_prec = gk; % drive the gradient of f in xk_prec to zero
        % gk_prec = gprecond; this would give a different variant of Anderson
        % Acceleration, where the preconditioner step is driven to zero,
        % rather than the gradient of the objective function being driven
        % to zero (note: this could also be done for NGMRES)
        
        % STEP II: Compute the NGMRES accelerated iterate
        %-------------------------------------------------       
        % form the least-squares system efficiently by storing some previous
        % inner products; see Washio and Oosterlee, ETNA 6, pp. 271-290, 1997
        eta=gk'*gk;
        for i=1:curW % iterations in window
            ksi(i)=gk'*Gwindow(:,i);
            beta(i,1)=eta-ksi(i);
        end
        for i=1:curW % iterations in window
            for j=1:curW % iterations in window
                Mat(i,j)=GtGwind(i,j)-ksi(i)-ksi(j)+eta;
            end
        end
        delta=epsi*max(max(diag(Mat)),epsi);
        
        % solve the normal equation system
        alpha=(Mat(1:curW,1:curW)+delta*eye(size(Mat(1:curW,1:curW),1))) \ beta(1:curW);
        
        if isnan(norm(alpha))
            restart = true;
            xk = xkold;
            fk = fkold;
            gk = gkold;
        end
        
        % Compute the accelerated approximation
        coef = 1-sum(alpha);
        x_a = coef*xk;
        
        for w = 1:curW
            x_a = x_a+alpha(w)*Xwindow(:,w);
        end
        
        % STEP III: Perform a line search for globalization
        %-------------------------------------------------
        pk=x_a-xk; % Search directory for the line search
        % We'll do a restart if pk is not a descent direction:
        % (pk is a descent direction if dk * gk < 0)
        % (recall that gk is the gradient, the steepest ascent direction)
        if pk' * gk >= 0
            restart = true;
        end
        
        if (~restart)
            [xk,fk,gk,~,~,nfev] = poblano_linesearch(FUN,xk,fk,gk,1,pk,params.Results);
            % Use the Poblano toolbox linesearch method (cvsrch).
            % Note: It will normally return the input xk if pk is not a descent direction
        end
        
        % Provide some output and get some log information
        out = poblano_out(xk,fk,gk,nfev+nfev_precond,params,out);
         
        if curW < params.Results.W
            curW = curW+1;
        end
        
        if ~restart % No restart, so update X and G
            j = mod(cntLoc,params.Results.W)+1; % The column j of X and G to update
            if strcmp(params.Results.AlgorithmVariant, 'ngmres')
                Xwindow(:,j) = xk;
                Gwindow(:,j) = gk;
                for i=1:curW
                    GtGwind(j,i)=gk'*Gwindow(:,i);
                    GtGwind(i,j)=GtGwind(j,i);
                end
            else % for Anderson acceleration, use preconditioned iterate in the extrapolation formula
                Xwindow(:,j) = xk_prec;
                Gwindow(:,j) = gk_prec;
                for i=1:curW
                    GtGwind(j,i)=gk_prec'*Gwindow(:,i);
                    GtGwind(i,j)=GtGwind(j,i);
                end
            end
        end
    end
end

% Prepare the output data
if params.Results.TraceRestarts
    Restarts = Restarts(1:out.Iters);
    out.TraceRestarts = Restarts;
end

