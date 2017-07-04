%% Load data
clear
load tvar_data;
model.nx = 1 + model.P + 1; % For the purpose of storage; z/rho/phi

%% Bootstrap particle filter
par.Np = 100;
par.resampling = 1;
tic;
[X,w] = pf(y, model, par);
toc;

plot(rho_true(1,1:T),'k-','linewidth',2);
hold on;
plot(squeeze(X(2,:,:))','r--');
hold off;
drawnow;

%% Standard Particle Gibbs / PGAS
par.Np = 50;
par.numMCMC = 50;
par.resampling = 1;
par.as = 1; % 0=PGibbs, 1=PGAS

% Compute ESJD
ESJD_aggr = zeros(model.nx,model.nx,T);

[X,X0] = cpf(y, model, par);

updts = zeros(1,T);
for(r = 1:par.numMCMC)
    [Xtmp,X0] = cpf(y, model, par, X, X0);
    updts = updts + all(Xtmp~=X,1);
    
    % Compute ESJD
    diff = Xtmp-X; % [nx,T]
    for(d = 1:model.nx); % Outer product for each t / loop over dimension instead of T
        ESJD_aggr(:,d,:) = ESJD_aggr(:,d,:) + reshape(bsxfun(@times, diff, diff(d,:)),[model.nx,1,T]);
    end    
    X = Xtmp;
    
    figure(1);
    plot(squeeze(ESJD_aggr(2,2,:))/r,'k--');    
    
    figure(2)
    plot(updts/r);
    title(sprintf('Iteration %i',r'));    
    drawnow;
end

%% Block PGibbs
par.Np = 50;
par.numMCMC = 50;
par.blockSize = 50;
par.overlap = 10;
par.resampling = 1;
[X,X0] = cpf(y, model, par);

% Compute ESJD
ESJD_aggr = zeros(model.nx,model.nx,T);

updts = zeros(1,T);
for(r = 1:par.numMCMC)
    [Xtmp,X0] = RLsampler(y, model, par, X, X0); % RL blocking
    %[Xtmp,X0] = PARsampler(y, model, par, X, X0); % PAR blocking
    updts = updts + all(Xtmp~=X,1);
    % Compute ESJD
    diff = Xtmp-X; % [nx,T]
    for(d = 1:model.nx); % Outer product for each t / loop over dimension instead of T
        ESJD_aggr(:,d,:) = ESJD_aggr(:,d,:) + reshape(bsxfun(@times, diff, diff(d,:)),[model.nx,1,T]);
    end    
    X = Xtmp;
    
   figure(3);
   plot(squeeze(ESJD_aggr(2,2,:))/r,'k--');
     
    figure(4)
    plot(updts/r);
    title(sprintf('Iteration %i',r'));    
    drawnow;
end
toc;
