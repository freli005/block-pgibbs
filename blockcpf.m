function [Xp,X0] = blockcpf(y, model, par, B, X, X0)
% Block conditional PF for the TVAR model
% y : [1 u-s+1] observations (for block)
% model : specifiec the model parameters
% par : specifies various tuning parameters
% B : [s,u], start and end index of block
% X : [P+2 u'-s+1] conditional state trajectory for t=s,...,u' where
%      u' = min(u+P, L) to account for the P-Markov dependencies and
%      boundary effects
% X0 : [2P+1] or conditional "initial" state, for time point x_{u-1} (which
%      could be 0)

par.resampling = 1; % Only allow multinomial for PGibbs
Np = par.Np;
L = length(y);
Lp = size(X,2); % Length of conditioned path (could be up to L+P)
if(L~=B(2)-B(1)+1 || Lp > L+model.P), error('BCPF: Inconsistent block length'); end;

% Extract model parameters
P = model.P;
alpha = model.alpha;
beta = model.beta;
sigma_a = model.sigma_a;
sigma_phi = model.sigma_phi;
sigma_v = model.sigma_v;

% Allocate memory for particles
z = zeros(1,Np,L);   % signal (state)
rho = zeros(P,Np,L); % lattice coeff (state)
phi = zeros(1,Np,L); % log-std of process noise (state)
a = zeros(1,Np,L);   % ancestors
w = zeros(1,Np,L);   % weights
logW = zeros(1,Np);  % Intermediate

% Extract conditional paths
Z = X(1,:);
RHO = X(2:(P+1),:);
PHI = X(P+2,:);
AR = zeros(P,Np);
for(t = 1:Lp)
    ar_tmp = rc2poly(RHO(:,t));
    AR(:,t) = -ar_tmp(2:end);
end

%%% Note that we always sample x0 as well if u = 1!
if(B(1)==1) % Initial block
    % Initial state (time 0, no measurement here!)
    done = false(P*(Np-1),1); % Use a truncated zero-mean normal prior for rho0
    rho0 = ones(P*(Np-1),1);
    while(~all(done))
        rho0(~done) = sqrt(model.Sigma0(2))*randn(sum(~done),1);
        done(~done) = abs(rho0(~done)) < 1;
    end
    x0 = zeros(2*P+1,Np);
    x0(1:P,1:Np-1) = sqrt(model.Sigma0(1))*randn(P,Np-1); % z0
    x0((P+1):(2*P),1:Np-1) = reshape(rho0,[P,Np-1]); % rho0
    x0(2*P+1,1:Np-1) = sqrt(model.Sigma0(3))*randn(1,Np-1); % phi0
    % Set the initial particle according to the conditioning
    x0(:,Np) = X0;

    % Use temp variables for simpler updates
    % z_now = [z_t, ..., z_{t-P+1}] --- note that 
    % x0 = [z_{0}, ..., z_{0-P+1}, \rho_{0}, \phi_{0}], i.e. the z's
    % are in reverse temporal order in the first P components (doesn't matter
    % with iid prior, but affects parent block sampler)
    z_now = x0(1:P,:);
    rho_now = x0((P+1):(2*P), :);
    phi_now = x0(2*P+1, :);
else % Interior block; use x_{u-1} as initial state for all particles
    z_now = X0(1:P)*ones(1,Np);
    rho_now = X0((P+1):(2*P))*ones(1,Np);
    phi_now = X0(2*P+1)*ones(1,Np); 
end

%%% Loop
for(t = 1:L) % Block length (t = index into block)
    if(t ~= 1)
        ind = resampling(w(1,:,t-1), par.resampling);
        ind = ind(randperm(Np));
        ind(Np) = Np; % Conditioning (only "standard PGibbs")
        a(:,:,t) = ind;        
        
        %%% Resample
        z_now = z_now(:,ind); % z_now contains the lag-P signal
        rho_now = rho_now(:,ind);
        phi_now = phi_now(1,ind);
    end

    %%%% Sample from proposal / compute weights
    % Update reflection coefficients; independent truncated normals
    done = false(P*Np,1);
    rho_new = zeros(P*Np,1);
    rho_now = rho_now(:);
    tmp_counter = 0;
    while(~all(done))
        tmp_counter = tmp_counter+1;
        rho_new(~done) = beta*rho_now(~done) + sigma_a*randn(sum(~done),1);
        done(~done) = abs(rho_new(~done)) < 1;
    end
    rho(:,:,t) = reshape(rho_new,P,Np);
    
    % Update noise variance parameter
    phi(1,:,t) = alpha*phi_now + sigma_phi*randn(1,Np);

    % Set according to conditioning. Note that W only depends on rho, phi,
    % and old z-values, so the weight computation (in the loop below) will
    % be correct, even though we conditino on z /after/ the weight is
    % computed
    rho(:,Np,t) = RHO(:,t);
    phi(1,Np,t) = PHI(1,t);
    
    % Update signal and compute weights
    lambda = 1/sigma_v^2 + 1./exp(2*phi(1,:,t));
    for(i = 1:Np)
        %[~, ar_now] = latc2tf(rho(:,i,t), 'allpole'); % a_now(1) = 1
        ar_now = rc2poly(rho(:,i,t)); % Should be the same as above!
        mu = -ar_now(2:end)*z_now(:,i);
        zhat = 1/lambda(i)*(y(t)/sigma_v^2 + mu/exp(2*phi(1,i,t)));
        tmpvar = sigma_v^2 + exp(2*phi(1,i,t));
        logW(i) = -1/2*log( tmpvar ) - 1/(2*tmpvar)*(y(t)-mu)^2;
        z(1,i,t) = zhat + 1/sqrt(lambda(i))*randn(1);        
    end
    
    % Set new z according to conditioning
    z(1,Np,t) = X(1,t);
    
    % Keep track of signal P steps, and other current states (used in AS
    % step)
    z_now = [z(1,:,t) ; z_now(1:end-1,:)];
    rho_now = rho(:,:,t);
    phi_now = phi(1,:,t);
    
    % Normalize weights
    const = max(logW);
    W = exp(logW-const);    
    if(const == -Inf), error('BCPF: Weights = 0!'); end;
    W = W/sum(W);
    w(1,:,t) = W; % Save the weights    
end

%%%% Add conditioning on future reference trajectory; note that this is
% only done at the last iteration of the block, even though z_t would
% affect future z's for t>s-P, though this dependence will be difficult to
% take into account. However, it is sufficient to add all the factors at
% the last iteration for correctness!

if(Lp > L) % If /not/ end block
    %%% Contribution from the Z's
    for(s = L+1:Lp)
        diff_tmp = Z(s)-AR(:,s)'*z_now; % [1,N]
        logW = logW - 1/2*exp(-2*PHI(s))*diff_tmp.^2;
        % Keep track of signal P steps
        z_now = [Z(s)*ones(1,Np) ; z_now(1:end-1,:)];
    end
    %%% Contribution from RHO; we have a truncated normal
    % transition for this state variable (iid over components)
    % which means that we get a particle-dependentn normalization
    % (note that we assume that the reference trajectory is in
    % the support of the prior here!)
    %
    % First the unnormalized (log-)Gaussian
    logW = logW - 1/(2*sigma_a^2)*sum(bsxfun(@minus, RHO(:,L+1), beta*rho_now).^2,1);
    % ...then the contributions from normalization
    upper_prb = normcdf(1,beta*rho_now, sigma_a); % [P,N]
    lower_prb = normcdf(-1,beta*rho_now, sigma_a);
    logW = logW - sum(log(upper_prb-lower_prb),1);
    %%% Contribution from PHI
    logW = logW - 1/(2*sigma_phi^2)*(PHI(L+1)-alpha*phi_now).^2;    
    %%% Normalize and sample
    const = max(logW);
    W = exp(logW-const);
    if(const == -Inf), error('BCPF: Weights = 0!'); end;
    W = W/sum(W);
    
    w(1,:,L) = W; % Save the weights        
end

%%%% Generate output
% Sampled trajectory
Xp = zeros(P+2,L);
ind = catrnd(w(1,:,L));
Xp(1,L) = z(1,ind,L);
Xp(2:(P+1),L) = rho(:,ind,L);
Xp(P+2,L) = phi(1,ind,L);

for(t = L-1:-1:1)
    ind = a(1,ind,t+1);
    Xp(1,t) = z(1,ind,t);
    Xp(2:(P+1),t) = rho(:,ind,t);
    Xp(P+2,t) = phi(1,ind,t);
end

if(B(1)==1) % If initial block, then we also update X0
    % N.B. No resampling at first step, so "ind(0) = ind(1)"
    X0 = x0(:, ind);
end