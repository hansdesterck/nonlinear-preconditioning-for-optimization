function out = lbfgs_prec(FUN,x0,varargin)
%LBFGS   Limited-memory BFGS minimization (vector-based).
%
%  OUT = LBFGS(FUN,X0) minimizes FUN starting at the point X0 using L-BFGS.
%  FUN is a handle for a function that takes a single vector input and
%  returns two arguments --- the scalar function value and the
%  vector-valued gradient. See POBLANO_OUT for details of the output
%  parameters.
%
%  OUT = LBFGS(FUN,X0,'param',value,...) specifies a parameters and its
%  value. See POBLANO_PARAMS for further details on standard parameters.
%  Additionally, LBFGS requires
%
%  'M' - Limited memory parameter {5}.
%
%  PARAMS = LBFGS('defaults') returns a structure containing the
%  default parameters for the particular Poblano method.
%
%  LBFGS can optionally be used with a nonlinear preconditioner (see
%     https://doi.org/10.1002/nla.2202 ), using optional parameters:
%
%  'PrecondType' - type of nonlinear preconditioning {'none'}
%       'none' - no nonlinear preconditioning
%       'LP' - nonlinear left-preconditioning
%       'TP' - nonlinear transformation preconditioning
%
%  'FUNPrecond' - nonlinear preconditioner function handle {@(x) x}
%
%  'LS_precond' - line search method when nonlinear preconditioning is used {'modBT'}
%       'modBT' - a specialised line search method developed for
%       nonlinearly preconditioned LBFGS
%       'more-thuente' - the standard more-thuente line search
%       (Note: LS_precond overrides the general parameter LineSearch_method
%       when nonlinear preconditioning is specified for LBFGS)
%
%
%  Examples
%
%  Suppose the function and gradient of the objective function are
%  specified in an mfile named example1.m:
%
%    function [f,g]=example1(x,a)
%    if nargin < 2, a = 1; end
%    f = sin(a*x);
%    g = a*cos(a*x);
%
%  We can call the optimization method (using its default
%  parameters) using the command:
%
%    out = lbfgs(@(x) example1(x,3), pi/4);
%
%  To change a parameter, we can specify a param/value input pair
%  as follows:
%
%    out = lbfgs(@(x) example1(x,3), pi/4, 'Display', 'final');
%
%  Alternatively, we can use a structure to define the parameters:
%
%    params.MaxIters = 2;
%    out = lbfgs(@(x) example1(x,3), pi/4, params);
%
%  See also POBLANO_OUT, POBLANO_PARAMS, FUNCTION_HANDLE.
%
%MATLAB Poblano Toolbox.
%Copyright 2009-2012, Sandia Corporation.

%% Parse parameters

% Create parser
params = inputParser;

% Set Poblano parameters
params = poblano_params(params);

% Set parameters for this method
params.addParamValue('M',5,@(x) x > 0);
params.addParamValue('PrecondType','none', @(x) ismember(x,{'none','LP','TP'}));
params.addParamValue('FUNPrecond',@(x) x, @(x) isa(x, 'function_handle'));
params.addParamValue('LS_precond','modBT',@(x) ismember(x,{'more-thuente','modBT'}));

% Parse input
params.parse(varargin{:});

%% Check input arguments
if (nargin == 1) && isequal(FUN,'defaults') && (nargout == 1)
    out = params.Results;
    return;
elseif (nargin < 2)
    error('Error: invalid input arguments');
end

%% Initialize

xk = x0;
if (strcmp(params.Results.PrecondType, 'none'))
    [fk,gk] = feval(FUN,xk);
    out = poblano_out(xk,fk,gk,1,params);
else
    [fk,gk_orig] = feval(FUN,xk);
    [gk,npcfev] = feval(params.Results.FUNPrecond,xk);
    out = poblano_out(xk,fk,gk_orig,1+npcfev,params);
end

%% Main loop
while out.ExitFlag == -1

    if (strcmp(params.Results.PrecondType, 'none'))
        % standard L-BFGS using the two-loop recursion
        if out.Iters == 0
            % Initialize quantities before first iteration
            pk = -gk;
            ak = 1.0;
            S = [];
            Y = [];
            rho = [];
        else
            % Precompute quantites used in this iteration
            sk = xk - xkold;
            yk = gk - gkold;
            skyk = yk'*sk;
            ykyk = yk'*yk;
            rhok = 1 / skyk;
            gamma = skyk/ykyk;

            % Use information from last M iterations only
            if out.Iters <= params.Results.M
                S = [sk S];
                Y = [yk Y];
                rho = [rhok rho];
            else
                S = [sk S(:,1:end-1)];
                Y = [yk Y(:,1:end-1)];
                rho = [rhok rho(1:end-1)];
            end

            % Adjust M to available number of iterations
            m = size(S,2);

            % L-BFGS two-loop recursion
            q = gk;
            for i = 1:m
                alpha(i) = rho(i)*S(:,i)'*q;
                q = q - alpha(i)*Y(:,i);
            end
            r = gamma*q;
            for i = m:-1:1
                beta = rho(i)*Y(:,i)'*r;
                r = r + (alpha(i) - beta)*S(:,i);
            end

            % r contains H_k * g_k (Hessian approximation at iteration k times
            % the gradient at iteration k
            pk = -r;
        end
        xkold = xk;
        gkold = gk;

        % Compute step length
        [xk,fk,gk,ak,lsinfo,nfev] = poblano_linesearch(FUN,xk,fk,gk,ak,pk,params.Results);
        if (lsinfo ~= 1) && strcmp(params.Results.Display, 'iter')
            fprintf(1,[mfilename,': line search warning = %d\n'],lsinfo);
        end

        % Update counts, check exit conditions, etc.
        out = poblano_out(xk,fk,gk,nfev,params,out);

    else
        % nonlinearly preconditioned L-BFGS
        if out.Iters == 0
            % Initialize quantities before first iteration
            pk = -gk;
            ak = 1.0;
            S = [];
            Y = [];
            Z = [];
            D = [];
            R = [];
        else
            % Precompute quantites used in this iteration
            sk = xk - xkold;
            yk = gk - gkold;
            zk = gk_orig - gk_origold;

            % The following modification is based on Procedure 18.2 (Damped 
            % BFGS Updating) from "Numerical Optimization, Second Edition" by
            % Nocedal and Wright. The purpose is to ensure that the update is
            % always well defined by modifying the definition of yk, which was
            % found to increase robustness for tensor decomposition problems.
            % theta=1 corresponds to the standard BFGS method.
            sTy = sk'*yk;
            if sTy < 0
                L = tril(S'*Y,-1);

                if isempty(S)
                    Bs = sk;
                    sBs = sk'*sk;
                else
                    delta = (Y(:,end)'*Y(:,end))/(S(:,end)'*Y(:,end));
                    dS = delta*S;
                    Bs = delta*sk - ([dS Y]*([S'*dS L; L' -D]\[dS'; Y']))*sk;
                    sBs = sk'*Bs;
                end

                if sTy >= 0.1*sBs
                    theta = 1;
                else
                    theta = (0.9*sBs)/(sBs-sTy);
                end

                yk = theta*yk + (1-theta)*Bs;
            end

            % Use information from last M iterations only
            idx = size(S,2);
            if idx < params.Results.M
                S = [S sk];
                Y = [Y yk];
                Z = [Z zk];
                if strcmp(params.Results.PrecondType, 'TP')
                    D(idx+1,idx+1) = zk'*sk;
                    R = [[R; zeros(1,idx)] S'*zk];
                else
                    D(idx+1,idx+1) = yk'*sk;
                    R = [[R; zeros(1,idx)] S'*yk];
                end
            else
                S = [S(:,2:end) sk];
                Y = [Y(:,2:end) yk];
                Z = [Z(:,2:end) zk];
                D = diag(D);
                R = R(2:end,2:end);
                if strcmp(params.Results.PrecondType, 'TP')
                    D = diag([D(2:end);sk'*zk]);
                    R = [[R; zeros(1,params.Results.M-1)] S'*zk];
                else
                    D = diag([D(2:end);sk'*yk]);
                    R = [[R; zeros(1,params.Results.M-1)] S'*yk];
                end
            end

            % Update below replaces the 2-loop recursion, which is impractical
            % when using nonlinear preconditioning. This update is based on the
            % compact BFGS representation of equation (3.1) in "Representations
            % of quasi-Newton matrices and their use in limited memory methods"
            % by Byrd, Nocedal, and Schnabel. For standard (non-preconditioned)
            % LBFGS, this has the same cost as the 2-loop recursion.
            if strcmp(params.Results.PrecondType, 'TP')
                gamma = (S(:,end)'*Z(:,end))/(Z(:,end)'*Y(:,end));
                r = gamma*gk;
                gY = gamma*Y;
                Mtmp = [S gY]'*gk_orig;
                Mtmp = [R'\(D + gY'*Z)/R, -eye(size(R))/(R'); -eye(size(R))/R, zeros(size(R))]*Mtmp;
                r = r + [S gY]*Mtmp;
            else
                gamma = (S(:,end)'*Y(:,end))/(Y(:,end)'*Y(:,end));
                r = gamma*gk;
                gY = gamma*Y;
                Mtmp = [S gY]'*gk;
                Mtmp = [R'\(D + Y'*gY)/R, -eye(size(R))/(R'); -eye(size(R))/R, zeros(size(R))]*Mtmp;
                r = r + [S gY]*Mtmp;
            end

            % r contains H_k * g_k (Hessian approximation at iteration k times
            % the gradient at iteration k
            pk = -r;
        end
        xkold = xk;
        fkold = fk;
        gkold = gk;
        gk_origold = gk_orig;

        % Execute linesearch.
        if strcmp(params.Results.LS_precond,'modBT')
           % Use the modified backtracking linesearch method, hardcoded here as it
           % is only intended for use with nonlinear preconditioning. See the paper
           % "Nonlinearly Preconditioned L-BFGS as an Acceleration Mechanism for
           % Alternating Least Squares, with Application to Tensor Decomposition" by 
           % De Sterck and Howse for details (Numer Linear Algebra Appl. 2018;e2202;
           % https://doi.org/10.1002/nla.2202)
            ak = 1;
            nfev = 1;
            xk = xk+ak*pk;
            [fk,gk_orig] = feval(FUN,xk);

            dec = 0;
            while (fk > (1+exp(-2*out.Iters))*fkold && dec < 4)
                dec = dec + 1;
                if dec == 3
                    S = [];
                    Y = [];
                    Z = [];
                    D = [];
                    R = [];
                    ak = 2;
                    pk = -gk;
                end
                xk = xkold;
                ak = ak*0.5;
                nfev = nfev+1;
                xk = xk+ak*pk;
                [fk,gk_orig] = feval(FUN,xk);
            end
        else
           % Use the original Poblano toolbox cvsrch linesearch method
            [newxk,newfk,newgk_orig,ak,~,nfev] = poblano_linesearch(FUN,xk,fk,gk_orig,ak,pk,params.Results);

           % For the case of nonlinear preconditioning, we add a check for the
           % sake of robustness that restarts the method with a steepest descent
           % step if issues are encountered.
            if (newfk >= fk) || isnan(newfk)
                ak = 1e-0;
                pk = -gk;
                if strcmp(params.Results.PrecondType, 'LP')
                    Y = [];
                    Z = [];
                    S = [];
                    R = [];
                    D = [];
                end
                newxk = xk+ak*pk;
                [newfk,newgk_orig] = feval(FUN,newxk);
                nfev = nfev + 1;
            end

            xk = newxk;
            fk = newfk;
            gk_orig = newgk_orig;
        end

        [gk,npcfev] = feval(params.Results.FUNPrecond,xk);

        % Update counts, check exit conditions, etc.
        out = poblano_out(xk,fk,gk_orig,nfev+npcfev,params,out);

    end
end
