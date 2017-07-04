function [X,X0] = cpf(y, model, par, X, X0)
% Particle filter (conditional or nonconditional) for the TVAR model
% X : [P+2 T] conditional state trajectory for t=1,...,T
% X0 : [2P+1] conditional initial state

conditioning = exist('X','var');
if(conditioning)
    par.resampling = 1; % Only allow multinomial for PGibbs
end
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

if(conditioning)    
    % Extract conditional paths
    Z = X(1,:);
    RHO = X(2:(P+1),:);
    PHI = X(P+2,:);
    AR = zeros(P,Np);
    for(t = 1:T)
        ar_tmp = rc2poly(RHO(:,t)); 
        AR(:,t) = -ar_tmp(2:end);
    end
end

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
if(conditioning)
    x0(:,Np) = X0; % Set the initial particle according to the conditioning
end

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
        ind = ind(randperm(Np));

        %%%% Ancestor sampling
        if(conditioning)
            if(par.as == 0) % Standard PGibbs
                ind(Np) = Np;
            elseif(par.as == 1) % Ancestor sampling w/o rejuvenation
                %%%% Contribution from the Z's
                logASW = zeros(1,Np);
                z_now_tmp = z_now; % Copy the lag-P z-particles [P,N]
                for(s = t:min(t+P-1,T))
                    diff_tmp = Z(s)-AR(:,s)'*z_now_tmp; % [1,N]
                    logASW = logASW - 1/2*exp(-2*PHI(s))*diff_tmp.^2;
                    % Keep track of signal P steps
                    z_now_tmp = [Z(s)*ones(1,Np) ; z_now_tmp(1:end-1,:)];
                end
                %%%% Contribution from RHO; we have a truncated normal
                % transition for this state variable (iid over components)
                % which means that we get a particle-dependent normalization
                % (note that we assume that the reference trajectory is in
                % the support of the prior here!)
                % 
                % First the unnormalized (log-)Gaussian
                logASW = logASW - 1/(2*sigma_a^2)*sum(bsxfun(@minus, RHO(:,t), beta*rho_now).^2,1);
                % ...then the contributions from normalization
                upper_prb = normcdf(1,beta*rho_now, sigma_a); % [P,N]
                lower_prb = normcdf(-1,beta*rho_now, sigma_a);
                logASW = logASW - sum(log(upper_prb-lower_prb),1);
                %%%% Contribution from PHI
                logASW = logASW - 1/(2*sigma_phi^2)*(PHI(t)-alpha*phi_now).^2;
                %%%% Contribution from importance weights at previous iteration
                logASW = logASW + logW;
                %%%% Normalize and sample
                const = max(logASW);
                ASW = exp(logASW-const);    
                if(const == -Inf), error('CPF: AS weights = 0!'); end;
                ASW = ASW/sum(ASW);
                ind(Np) = catrnd(ASW);
            else % Ancestor sampling w rejuvenation
                error('Rejuvenation not implemented yet');
            end
        end
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
    if(conditioning)
        rho(:,Np,t) = RHO(:,t);
        phi(1,Np,t) = PHI(1,t);
    end
    
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
    
    % Set new z according to conditioning
    if(conditioning)
        z(1,Np,t) = X(1,t);
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
% Sampled trajectory
ind = catrnd(w(1,:,T));
X(1,T) = z(1,ind,T);
X(2:(P+1),T) = rho(:,ind,T);
X(P+2,T) = phi(1,ind,T);

for(t = T-1:-1:1)
    ind = a(1,ind,t+1);
    X(1,t) = z(1,ind,t);
    X(2:(P+1),t) = rho(:,ind,t);
    X(P+2,t) = phi(1,ind,t);
end
% N.B. No resampling at first step, so "ind(0) = ind(1)"
X0 = x0(:, ind);

% All trajectories
% w = w(:,:,T); % Only return the last weights

% % Generate the trajectories from ancestors
% ind = a(:,:,T);
% for(t = T-1:-1:1)
%     z(1,:,t) = z(1,ind,t);
%     rho(:,:,t) = rho(:,ind,t);
%     phi(1,:,t) = phi(1,ind,t);
%     ind = a(1,ind,t);
% end