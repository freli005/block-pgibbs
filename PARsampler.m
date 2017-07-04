function [X,X0] = PARsampler(y, model, par, X, X0)
% Runs one complete iteration of the parallelizable blocked PGibbs sampeler

T = length(y); % Number of samples
L = par.blockSize;
p = par.overlap;
P = model.P; % AR order

if(p>=L/2)
    error('Overlap cannot exceed block size');
end

inbetween = L-2*p;
odd_start = 1:(L+inbetween):T;
even_start = (L-p+1):(L+inbetween):T;
startid = [odd_start even_start];
numBlocks = length(startid);

for(j=1:numBlocks)
    s = startid(j);
    u = min(s+L-1,T);
    
    %%% Set up conditioning on the left
    if(s > 1) % Intermediate block
        if(s > P)
            Z0 = X(1,s-P:s-1)';
            Z0 = Z0(end:-1:1); % Most recent ancestor first in Z0!
        else % s \in [2, P]
            Z0 = X(1,1:s-1)';
            Z0 = Z0(end:-1:1); % Most recent ancestor first in Z0!
            Z0 = [Z0 ; X0(1:P-s+1)]; % Remaining P-s+1 conditional states contained in X0
        end
        X0c = [Z0 ; X(2:end,s-1)]; % rho and phi
    else % First block
        X0c = X0; 
    end
    
    %%% Set up conditioning internally and on the right
    Xc = X(:,s:min(u+P,T));
    
    %%% Run CPF and update block
    [Xp,X0p] = blockcpf_bridge(y(s:u), model, par, [s u], Xc, X0c);
    X(:,s:u) = Xp;
    if(s == 1) % Initial block, update X0
        X0 = X0p;
    end
end
    
    