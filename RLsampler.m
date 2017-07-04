function [X,X0] = RLsampler(y, model, par, X, X0)
% Runs one complete iteration of the right-to-left blocked PGibbs sampeler

T = length(y); % Number of samples
L = par.blockSize;
p = par.overlap;
P = model.P; % AR order

if(p>=L)
    error('Overlap cannot exceed block size');
end

% First (=rightmost) block coordinates
u = T; s = max(T-L+1,1);
done = false;
while(~done)
    done = (s==1); % If we reach s=1, then we run one more iteration of the loop and quit
    
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
    
    %%% Compute coordinates for next block
    u = u-L+p;
    s = max(s-L+p,1);
end
    
    