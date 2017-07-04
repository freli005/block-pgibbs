function [z,rho,phi,y] = simulator(T,model)
% TVAR model by Godsill et al. (2004)

% Extract model parameters
P = model.P;
alpha = model.alpha;
beta = model.beta;
sigma_a = model.sigma_a;
sigma_phi = model.sigma_phi;
sigma_v = model.sigma_v;

% Initial state
%nx = [2*P + 1]; % phi_{v_t} is not dynamic in this version of the model
%x0 = mvnrnd(model.mu0, model.Sigma0)';
done = false(P,1); % Use a truncated zero-mean normal prior for rho0
rho0 = ones(P,1);
while(~all(done))
    rho0(~done) = sqrt(model.Sigma0(2))*randn(sum(~done),1);
    done(~done) = abs(rho0(~done)) < 1;
end
x0 = zeros(2*P+1,1);
x0(1:P) = sqrt(model.Sigma0(1))*randn(P,1); % z0
x0((P+1):(2*P)) = rho0;
x0(2*P+1) = sqrt(model.Sigma0(3))*randn(1,1); % phi0

% Allocate memory
z = zeros(1,T);
rho = zeros(P,T);
phi = zeros(1,T);
y = zeros(1,T);

% Use temp variable for simpler updates
% Note that x0 = [z_{0}, ..., z_{0-P+1}, \rho_{0}, \phi_{0}], i.e. the z's
% are in reverse temporal order in the first P components (doesn't matter
% with iid prior, but still worth noting)
z_now = x0(1:P);
rho_now = x0((P+1):(2*P));
phi_now = x0(2*P+1);

for(t = 1:T)
    % Update reflection coefficients; independent truncated normals
    done = false(P,1);
    while(~all(done))
        rho(~done,t) = beta*rho_now(~done) + sigma_a*randn(sum(~done),1);
        done(~done) = abs(rho(~done,t)) < 1;
    end
    rho_now = rho(:,t);
    
    % Update noise variance parameter
    phi(t) = alpha*phi_now + sigma_phi*randn(1);
    phi_now = phi(t);
    
    % Update signal    
    [~, a_now] = latc2tf(rho_now, 'allpole'); % a_now(1) = 1
    z(t) = -a_now(2:end)*z_now + exp(phi_now)*randn(1);
    z_now = [z(t) ; z_now(1:end-1)];
    
    % Simulate measurement
    y(t) = z(t) + sigma_v*rand(1);
end
    