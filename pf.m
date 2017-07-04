function [X,w] = pf(y, model, par)
% Particle filter (nonconditional) for the TVAR model

Np = par.Np;
T = length(y);

% Extract model parameters
P = model.P;
alpha = model.alpha;
beta = model.beta;
sigma_a = model.sigma_a;
sigma_phi = model.sigma_phi;
sigma_v = model.sigma_v;

% Allocate memory for particles
z = zeros(1,Np,T);   % signal (state)
rho = zeros(P,Np,T); % lattice coeff (state)
phi = zeros(1,Np,T); % log-std of process noise (state)
a = zeros(1,Np,T);   % ancestors
w = zeros(1,Np,T);   % weights
logW = zeros(1,Np);  % Intermediate

% Initial state (time 0, no measurement here!)
done = false(P*Np,1); % Use a truncated zero-mean normal prior for rho0
rho0 = ones(P*Np,1);
while(~all(done))
    rho0(~done) = sqrt(model.Sigma0(2))*randn(sum(~done),1);
    done(~done) = abs(rho0(~done)) < 1;
end
x0 = zeros(2*P+1,Np);
x0(1:P,:) = sqrt(model.Sigma0(1))*randn(P,Np); % z0
x0((P+1):(2*P),:) = reshape(rho0,[P,Np]); % rho0
x0(2*P+1,:) = sqrt(model.Sigma0(3))*randn(1,Np); % phi0

% Use temp variables for simpler updates
% z_now = [z_t, ..., z_{t-P+1}] --- note that 
% x0 = [z_{0}, ..., z_{0-P+1}, \rho_{0}, \phi_{0}], i.e. the z's
% are in reverse temporal order in the first P components (doesn't matter
% with iid prior, but still worth noting)
z_now = x0(1:P,:);
rho_now = x0((P+1):(2*P), :);
phi_now = x0(2*P+1, :);

% Loop
for(t = 1:T)
    if(t ~= 1)
        ind = resampling(w(1,:,t-1), par.resampling);
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

    % Update signal and compute weights
    lambda = 1/sigma_v^2 + 1./exp(2*phi(1,:,t));
    for(i = 1:Np)
        %[~, ar_now] = latc2tf(rho(:,i,t), 'allpole'); % a_now(1) = 1
        ar_now = rc2poly(rho(:,i,t)); % Should be the same as above!
        mu = -ar_now(2:end)*z_now(:,i);
        zhat = 1/lambda(i)*(y(t)/sigma_v^2 + mu/exp(2*phi(1,i,t)));
        tmpvar = sigma_v^2 + exp(2*phi(1,i,t));
        logW(i) = -1/2*log(tmpvar) - 1/(2*tmpvar)*(y(t)-mu)^2;
        z(1,i,t) = zhat + 1/sqrt(lambda(i))*randn(1);        
    end
    
    % Keep track of signal P steps, and other current states (used in AS
    % step)
    z_now = [z(1,:,t) ; z_now(1:end-1,:)];
    rho_now = rho(:,:,t);
    phi_now = phi(1,:,t);
    
    % Normalize weights
    const = max(logW);
    W = exp(logW-const);    
    if(const == -Inf), error('CPF: Weights = 0!'); end;
    W = W/sum(W);
    w(1,:,t) = W; % Save the weights    
end

%%%% Generate output
w = w(:,:,T); % Only return the last weights

% Generate the trajectories from ancestors
ind = 1:Np;
X = zeros(P+2,Np,T);
for(t = T:-1:1)
    X(1,:,t) = z(1,ind,t);
    X(2:(P+1),:,t) = rho(:,ind,t);
    X(end,:,t) = phi(1,ind,t);
    ind = a(1,ind,t);
end