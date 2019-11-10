function out = ncg_prec(FUN,x0,varargin)
%NCG   Nonlinear conjugate gradient minimization.
%
%  OUT = NCG(FUN,X0) minimizes FUN starting at the point
%  X0 using nonlinear conjugate gradients. FUN is a handle for a
%  function that takes a single vector input and returns two arguments
%  --- the scalar function value and the vector-valued gradient.
%  See POBLANO_OUT for details of the output parameters.
%
%  OUT = NCG(FUN,X0,'param',value,...) specifies a
%  parameters and its value. See POBLANO_PARAMS for further details on
%  standard parameters. Additionally, POBLANO_NCG requires
%
%  'Update' - conjugate direction update {'PR'}
%    'FR' Fletcher-Reeves NCG
%    'PR' Polak-Ribiere NCG
%    'HS' Hestenes-Stiefel NCG
%    'SD' Steepest Decsent
%
%  'RestartIters' - number of iterations to run before conjugate direction
%                   restart {20}
%
%  'RestartNW' - flag to use restart heuristic of Nocedal and Wright {false}
%
%  'RestartNWTol' - tolerance for Nocedal and Wright restart heuristic {0.1}
%
%  PARAMS = NCG('defaults') returns a structure containing the
%  default parameters for the particular Poblano method.
%
%  NCG can optionally be used with a nonlinear preconditioner (see
%     https://doi.org/10.1002/nla.2202), using optional parameters:
%
%  'UsePrecond' - use nonlinear preconditioner {false}
%
%  'FUNPrecond' - nonlinear preconditioner function handle {@(x) x}
%
%  'Update' - conjugate direction update
%    options that can be used for nonlinear preconditioning include:
%    'FR_LP','PR_LP','HS_LP','FR_TP','PR_TP','HS_TP'
%    (see https://doi.org/10.1002/nla.2202)
%
%
%  Examples
%
%  Suppose the function and gradient of the objective function are
%  specified in an mfile named mysin.m:
%
%    function [f,g]=example1(x,a)
%    if nargin < 2, a = 1; end
%    f = sin(a*x);
%    g = a*cos(a*x);
%
%  We can call the optimization method (using its default
%  parameters) using the command:
%
%    out = ncg(@(x) example1(x,3), pi/4);
%
%  To change a parameter, we can specify a param/value input pair
%  as follows:
%
%    out = ncg(@(x) example1(x,3), pi/4, 'Display', 'final');
%
%  Alternatively, we can use a structure to define the parameters:
%
%    params.MaxIters = 2;
%    out = ncg(@(x) example1(x,3), pi/4, params);
%
%  See also POBLANO_OUT, POBLANO_PARAMS, POBLANO_LINESEARCH, FUNCTION_HANDLE.
%
%MATLAB Poblano Toolbox.
%Copyright 2009-2012, Sandia Corporation.

%% Parse parameters

% Create parser
params = inputParser;

% Set Poblano parameters
params = poblano_params(params);

% Set parameters for this method
params.addParamValue('RestartIters',20,@(x) x > 0);
params.addParamValue('Update','PR',@(x) ismember(x,...
    {'FR','PR','HS','SD','FR_LP','PR_LP','HS_LP','FR_TP','PR_TP','HS_TP'}));
params.addParamValue('RestartNW',false,@islogical);
params.addParamValue('RestartNWTol',0.1,@(x) x > 0);
params.addParamValue('UsePrecond',false,@islogical);
params.addParamValue('FUNPrecond',@(x) x, @(x) isa(x, 'function_handle'));

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
if (~params.Results.UsePrecond)
    [fk,gk] = feval(FUN,xk);
    out = poblano_out(xk,fk,gk,1,params);
else
    [fk,gk_orig] = feval(FUN,xk);
    [gk,npcfev] = feval(params.Results.FUNPrecond,xk);
    out = poblano_out(xk,fk,gk_orig,1+npcfev,params);
end

%% Main loop
while out.ExitFlag == -1

    if (~params.Results.UsePrecond)
        if out.Iters == 0
            pk = -gk;
            ak = 1.0;
            gkTgk = gk'*gk;
        else
            % Compute next direction
            if mod(out.Iters,params.Results.RestartIters) == 0
                % restart to prevent stagnation
                bk = 0; %#ok<*NASGU>
                pk = -gk;
            else
                % direction update
                switch (params.Results.Update)
                    case 'FR'
                        % Fletcher-Reeves
                        gkTgk = gk'*gk;
                        if gkTgkold > 0
                            bk = gkTgk/gkTgkold;
                        else
                            fprintf(1,[mfilename,': warning: bk set to 0\n']);
                            bk = 0;
                        end
                    case 'PR'
                        % Polak-Ribiere
                        gkTgk = gk'*gk;
                        gkMgkold = gk-gkold;
                        if gkTgkold > 0
                            bk = (gk'*gkMgkold)/gkTgkold;
                        else
                            fprintf(1,[mfilename,': warning: bk set to 0\n']);
                            bk = 0;
                        end
                    case 'HS'
                        % Hestenes-Stiefel
                        gkMgkold = gk-gkold;
                        denom = pkold'*gkMgkold;
                        if denom > 0
                            bk = (gk'*gkMgkold)/denom;
                        else
                            fprintf(1,[mfilename,': warning: bk set to 0\n']);
                            bk = 0;
                        end
                    case 'SD'
                        % Steepest Descent
                        bk = 0;
                    otherwise
                        error('Error: options.Update is not valid. Choices are {FR, PR, HS}');
                end
                % do not allow negative conjugate direction weights
                if bk < 0
                    bk = max(0,bk);
                end

                % restart method from Nocedal and Wright
                if params.Results.RestartNW
                    v = params.Results.RestartNWTol;
                    if ((gk'*gkold)/(gkTgkold^2) >= v)
                        bk = 0;
                    end
                end

                % new direction
                pk = -gk + bk*pkold;
            end
        end

        gkold = gk;
        pkold = pk;
        gkTgkold = gkTgk;

        % Compute step length
        [xk,fk,gk,ak,lsinfo,nfev] = poblano_linesearch(FUN,xk,fk,gk,ak,pk,params.Results);
        if (lsinfo ~= 1) && strcmp(params.Results.Display, 'iter')
            fprintf(1,[mfilename,': line search warning = %d\n'],lsinfo);
        end

        % Update counts, check exit conditions, etc.
        out = poblano_out(xk,fk,gk,nfev,params,out);
    else
        if out.Iters == 0
            pk = -gk;
            ak = 1.0;
            if ismember(params.Results.Update,{'FR_TP','PR_TP'})
                gkTgk = gk_orig'*gk;
            else
                gkTgk = gk'*gk;
            end
        else
            % Compute next direction
            if mod(out.Iters,params.Results.RestartIters) == 0
                % Restart to prevent stagnation
                bk = 0;
                pk = -gk;
            else
                % Direction update
                switch (params.Results.Update)
                    case {'FR_LP','FR_TP'}
                        % Fletcher-Reeves
                        if strcmp(params.Results.Update, 'FR_LP')
                            gkTgk = gk'*gk;
                        else
                            gkTgk = gk_orig'*gk;
                        end
                        if gkTgkold > 0
                            bk = gkTgk/gkTgkold;
                        else
                            % fprintf(1,[mfilename,': warning: bk set to 0\n']);
                            bk = 0;
                        end
                    case {'PR_LP','PR_TP'}
                        % Polak-Ribiere
                        if strcmp(params.Results.Update, 'PR_LP')
                            gkTgk = gk'*gk;
                            gkMgkold = gk-gkold;
                        else
                            gkTgk = gk_orig'*gk;
                            gkMgkold = gk_orig-gkold_orig;
                        end
                        if gkTgkold > 0
                            bk = (gk'*gkMgkold)/gkTgkold;
                        else
                            % fprintf(1,[mfilename,': warning: bk set to 0\n']);
                            bk = 0;
                        end
                    case {'HS_LP','HS_TP'}
                        % Hestenes-Stiefel
                        if strcmp(params.Results.Update, 'HS_LP')
                            gkMgkold = gk-gkold;
                        else
                            gkMgkold = gk_orig-gkold_orig;
                        end
                        denom = pkold'*gkMgkold;
                        if denom > 0
                            bk = (gk'*gkMgkold)/denom;
                        else
                            % fprintf(1,[mfilename,': warning: bk set to 0\n']);
                            bk = 0;
                        end
                    otherwise
                        error(['Error: options.Update is not valid.' ...
                            'Choices are {FR_LP, PR_LP, HS_LP, FR_TP, PR_TP, HS_TP}']);
                end
                % Do not allow negative conjugate direction weights
                if bk < 0
                    bk = max(0,bk);
                end

                % Restart method from Nocedal and Wright
                if params.Results.RestartNW
                    v = params.Results.RestartNWTol;
                    if ((gk'*gkold)/(gkTgkold^2) >= v)
                        bk = 0;
                    end
                end

                % New direction
                pk = -gk + bk*pkold;

            end
        end

        gkold = gk;
        gkold_orig = gk_orig;
        pkold = pk;
        gkTgkold = gkTgk;

        % Use the Poblano toolbox linesearch method (cvsrch)
        [xk,fk,gk_orig,ak,lsinfo,nfev] = poblano_linesearch(FUN,xk,fk,gk_orig,ak,pk,params.Results);
        if (lsinfo ~= 1) && strcmp(params.Results.Display, 'iter')
            fprintf(1,[mfilename,': line search warning = %d\n'],lsinfo);
        end

        [gknew, npcfev] = feval(params.Results.FUNPrecond,xk);
        des = gk_orig'*gknew/norm(gknew,2)/norm(gk_orig,2);
        if des > 0
            gk = gknew;
        else
            gk = -gknew;
        end

        % Update counts, check exit conditions, etc.
        % Assume ALS is equivalent to 1 f and g evaluation
        out = poblano_out(xk,fk,gk_orig,nfev+npcfev,params,out);
    end
end
