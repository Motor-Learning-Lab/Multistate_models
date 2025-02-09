p = [zeros(80, 1); 30*ones(150, 1); zeros(80, 1)];

N = length(p);
zeroVec = zeros(N, 1);
x = zeroVec;
y = zeroVec;
epsilon = zeroVec;
sPHat = zeroVec;
pHat = zeroVec;
eta = zeroVec;

A = 0.98;
B = 0.2;

sEta = 0.4;
sPHat(1) = 0.4;
pHat(1) = normrnd(0,sPHat(1));

sEpsilon = 3;
pBroad = 45;
x(1) = normrnd(0, sEta(1));
epsilon(1) = normrnd(0, sEpsilon);
y(1) = x(1) + p(1) + epsilon(1);
eta(1) = normrnd(0, sEta(1));

for t = 2:N
    epsilon(t) = normrnd(0, sEpsilon);
    predErr(t) = pHat(t) - p(t) + epsilon(t);
    y(t) = x(t) + predErr(t); 

    sPHat(t) = sPUpdate(sPHat(t-1), sEpsilon, y(t), pBroad); % Shouldn't this be sBroad?
    
    eta(t) = normrnd(0, sEta);
    if t < N
        pHat(t+1) = pHatUpdate(pHat(t), y(t), sPHat(t), sEpsilon);
        x(t+1) = A*x(t) - B*y(t) + eta(t);
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
function sPNew = sPUpdate(sPHat, sEpsilon, y, sBroad)
underWeight = 0.4;
priorPChange = 0.05;
pChangeFn = @(p1,p2) (p2^underWeight*priorPChange)/(p2^underWeight*priorPChange + p1^underWeight*(1-priorPChange));

pPHat = normpdf(y, 0, sqrt(sPHat^2+sEpsilon^2));
pBroad = normpdf(y, 0, sqrt(sBroad^2+sEpsilon^2));
pChange = pChangeFn(pPHat, pBroad);

priorWeight = 500;
sPPosterior = sqrt( (priorWeight+1) / (1/sEpsilon^2 + priorWeight/sPHat^2) );
sPNew = pChange*sBroad + (1-pChange)*sPPosterior;
end


function pHatNew = pHatUpdate(pHat, y, pEta, sEpsilon)
precEta = 1/pEta^2;
precY = 1/sEpsilon^2;

pHatNew = pHat - y*precY / (precY + precEta);
end
