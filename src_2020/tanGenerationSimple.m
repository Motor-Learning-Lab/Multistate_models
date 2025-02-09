p = [zeros(80, 1); 30*ones(150, 1); zeros(80, 1)];

N = length(p);
zeroVec = zeros(N, 1);
x = zeroVec;
y = zeroVec;
epsilon = zeroVec;
eta = zeroVec;
sEta = zeroVec;
pEta = zeroVec;
P = zeroVec;
B = zeroVec;

A = 0.98;
B = 0.2;

sEta(1) = 0.4;
pEta(1) = 0.4;
sEpsilon = 3;
pBroad = 45;
x(1) = normrnd(0, sEta(1));
epsilon(1) = normrnd(0, sEpsilon);
y(1) = x(1) + epsilon(1);
[B(1), P(1)] = bUpdate(A, sEta(1), sEpsilon, sEta(1));
eta(1) = normrnd(0, sEta(1));

for t = 2:N
    epsilon(t) = normrnd(0, sEpsilon);
    y(t) = x(t) + p(t) + epsilon(t);

    pEta(t) = pEtaUpdate(pEta(t-1), sEpsilon, B(t-1), y(t), pBroad);
    eta(t) = normrnd(0, sEta(t));
    [B(t), P(t)] = bUpdate(A, pEta(t)+sEta(t), sEpsilon, P(t-1));
    if t < N
        x(t+1) = A*x(t) - B(t)*y(t) + eta(t);
    end
end


% nonNormalizedSurprise          CleverFunction
% 0         0                       0.95*sEta
% 0.34      0.5 SD                       
% 0.68      1                       1
% 0.96      2                       0.875*sEta + 0.125*max(Noise)
% 0.99      3                       0.75*sEta + 0.25*max(Noise)
%           4                       0.5(max(Noise)+sEta)
%           5                       max(Noise)
function pEtaNew = pEtaUpdate(pEta, sEta, sEpsilon, B, y, sBroad)
underWeight = 0.35;
priorPChange = 0.02;
pChangeFn = @(p1,p2) (p2^underWeight*priorPChange)/(p2^underWeight*priorPChange + p1^underWeight*(1-priorPChange));

pEta = normpdf(y, 0, pEta+sEta+sEpsilon);
pBroad = normpdf(y, 0, sBroad+sEta+sEpsilon);
pChange = pChangeFn(pEta, pBroad);
% pChange = abs(normcdf(y, 0, sEta+sEpsilon)-0.5)*2;
priorWeight = 10;
pEtaPosterior = sqrt( (priorWeight+1) / (1/(B*y)^2 + priorWeight/(pEta+sEta)^2) );
sEtaNew = pChange*sBroad + (1-pChange)*sEtaPosterior;
% if  pChange < 0.99
%     sEtaNew = pEtaPosterior;
% else
%     sEtaNew = sBroad;
% end

% errorKnots = [0 1 2 3 4 5]';
% newEtaKnots = [ ...
%     0.9*sEta ...
%     , 0.95*sEta ...
%     , 0.875*sEta + 0.125*sEpsilon ...
%     , 0.75*sEta + 0.25*sEpsilon ...
%     , 0.5*sEta + 0.5*sEpsilon ...
%     , sEpsilon ...
%     ];
% sEtaNew = interp1(errorKnots, newEtaKnots, normalizedError, 'linear', sEpsilon);
end


function [B, pNew] = bUpdate(A, sEta, sEpsilon, P)
pNew = A^2*(P-P^2/(P+sEpsilon^2))+sEta^2;
B = pNew / (sEpsilon^2 + pNew);
end
