clear;
%% Set up model
% Use the same model parameters as Godsill et al. (2004), with the
% exception that sigma_v = 1 and they use sigma_v = 0.02. Also, unclear
% what the initial state distribution is in their simulation.
model.P = 4;
model.alpha = 0.99;
model.beta = 1;
model.sigma_a = 0.01;
model.sigma_phi = 0.001;
model.sigma_v = 0.02;
%model.mu0 = [zeros(1,2*model.P),0];
%model.Sigma0 = blkdiag(.1*eye(2*model.P), model.sigma_phi^2/(1-model.alpha^2));
model.Sigma0 = [.1 .1 model.sigma_phi^2/(1-model.alpha^2)]; % Variance of z0 (iid), rho0 (iid) and phi0

%% Simulate some data
T = 2000;
[z_true,rho_true,phi_true,y] = simulator(T,model);

figure(1)
plot(rho_true')
figure(2);
plot(phi_true)
figure(3);
plot(z_true,'-');
hold on;
plot(y,'.');
hold off;

%% Compute initial trajectory (to be able to initialise close to posterior)
initpar.Np = 10000;
initpar.resampling = 3;
[Xinit,X0init] = cpf(y, model, initpar);

%%
save tvar_data