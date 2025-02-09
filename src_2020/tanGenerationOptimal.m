for t = 1:N
    epsilon = normrnd(0, sigmaEpsilon);
    y = x(t) + p(t) + epsilon(t);

    nonNormalizedSurprise(t) = 2*(normcdf(abs(y), 0, sigmaY)-0.5);
    surprise(t) = CleverFunction(nonNormalizedSurprise);
    sigmaEta(t) = surprise(t, sigmaEta(t-1));
    eta = normrnd(0, sigmaEta(t));
    x(t+1) = A*x(t) + B*y(t) + eta(t);
end

% nonNormalizedSurprise          CleverFunction
% 0         0                       0.95*sigmaEta
% 0.34      0.5 SD                       
% 0.68      1                       1
% 0.96      2                       0.875*sigmaEta + 0.125*max(Noise)
% 0.99      3                       0.75*sigmaEta + 0.25*max(Noise)
%           4                       0.5(max(Noise)+sigmaEta)
%           5                       max(Noise)