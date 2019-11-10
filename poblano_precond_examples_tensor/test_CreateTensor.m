% Hans De Sterck, June 2011
function [Zdprime,Z] = test_CreateTensor(params)
% params=[I collinearity R l1 l2]

% Create a test tensor
% (This is the test from Tomasi and Bro (2006) and Acar et al. (2011) for
% a ramdom dense tensor modified to get specified collinearity between the
% columns)

I = params(1);
collinearity = params(2);
R = params(3);
l1 = params(4);
l2 = params(5);

% First generate K
K = collinearity*ones(R,R)+(1-collinearity)*eye(R);

% Get C as the Cholesky factor of K
C = chol(K);

U = cell(3,1);
for n=1:3
    % Cenerate a random matrix
    M = randn(I,R);
    % Ortho-normalize the columns of M, gives Q
    [Q,~] = qr(M,0);    
    U{n} = Q*C;
end

Z = ktensor(U);
Zfull = full(Z);

% Generate two random tensors
N1 = tensor(randn(I,I,I));
N2 = tensor(randn(I,I,I));
nZ = norm(Zfull);
nN1 = norm(N1);

% Modify Z with the two different types of noise
Zprime = Zfull+1/sqrt(100/l1-1)*nZ/nN1*N1;
nZprime = norm(Zprime);

N2Zprime = N2.*Zprime;
nN2Zprime = norm(N2Zprime);

Zdprime = Zprime+1/sqrt(100/l2-1)*nZprime/nN2Zprime*N2Zprime;

end


