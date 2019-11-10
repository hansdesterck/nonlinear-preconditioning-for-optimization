function [g, nfev] = test_CPtensor_preconditioner(x,Z,r,ns,opt)

nf = 1;
nModes = ndims(Z);
xoriginal = x;

for i = 1:ns
    % Convert u to a ktensor
    A = ktensor(tt_cp_vec_to_fac(x,Z));

    % Perform one ALS iteration with the tensor A as input
    % If there is a lambda factor in A, put it into the first mode
    lambda = A.lambda;
    A.U{1} = A.U{1} * spdiags(lambda,0,r,r);
    A.lambda = ones(r,1);

    if strcmp(opt, 'TTB')
        % Use the tensor toolbox to carry out ALS
        A = cp_als(Z,r,'maxiters',1,'init',A.U,'printitn',0, 'fixsigns', false);
    else    
        % Iterate over all modes of the tensor
        for n = 1:nModes
            % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
            Unew = mttkrp(Z,A.U,n);

            % Compute the matrix of coefficients for linear system
            Y = ones(r,r);
            for j = [1:n-1,n+1:nModes]
                Y = Y .* (A.U{j}'*A.U{j});
            end

            Unew = (Y \ Unew')'; %<- Line from TTB 2.2.

            if issparse(Unew)
              A.U{n} = full(Unew);   % For the case r=1
            else
              A.U{n} = Unew;
            end
        end

        A = ktensor(A.U); % Default: lambda=1

        % Normalize and introduce lambda, and make sure the components appear in order of lambda
        A = arrange(A); 
    end
    
    % Distribute lambda evenly over all modes
    A = normalize(A,0);

    % Convert A to a vector
    x = tt_fac_to_vec(A.U);
end

g = xoriginal-x;
nfev = nf*ns;

end