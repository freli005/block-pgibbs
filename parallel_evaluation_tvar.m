function out = parallel_evaluation_tvar()
% This function runs simulations using parfor loops over different methods
%   to evaluate the block samplers on synthetic data

% Declaration for compiler
model = [];
y = [];
Xinit = [];
X0init = [];

% Load data
load('tvar_data','model','y','Xinit','X0init');

% Test save
pathname = './';
filename = 'tvar_results';
fprintf('Saving to %s%s\n',pathname,filename);
save([pathname filename],'-v7.3');

% Varying parameters
TVec = [1000 2000];
LpVec = [10 50 50 ; 0 0 10]; % each column is a pair L/p

% Fixed parameters
model.nx = 1 + model.P + 1; % For the purpose of storage; z/rho/phi
par.numMCMC = 500;
par.Np = 100; % Number of particles
par.resampling = 1; % Multinomial resampling

%% Create a "parameter table"

numT = length(TVec);
numLp = length(LpVec);

% The four columns are:
% sampler = 1,2,3 for cpf,RLsampler,PARsampler
% AS = 0,1,NaN for ancestor sampler off,on,N/A
% L = <blocksize>,NaN
% p = <overlap>,NaN
%     sampler  AS   L   p
partable = [2 NaN   1   0 ; % Gibbs
            1   0 NaN NaN ; % PGibbs
            1   1 NaN NaN]; % PGAS
partable = [partable;
    [2*ones(numLp,1) NaN(numLp,1) LpVec'];% RL sampler
    [3*ones(numLp,1) NaN(numLp,1) LpVec']];% PAR sampler
numMethods = size(partable,1);
tmp = [];
for(cT = 1:numT)
    tmp = [tmp; [TVec(cT)*ones(numMethods,1) partable]];
end
partable = tmp;
numSettings = size(partable,1); % Gibbs + PGibbs + PGAS + "RL & PAR for each L/p", for each T

%% Loop over all methods

parfor(id = 1:numSettings)
    % Extract parameters for current run
    T = partable(id,1);
    sampler = partable(id,2);
    
    par_now = par;
    par_now.as = partable(id,3);
    par_now.blockSize = partable(id,4);
    par_now.overlap = partable(id,5);
    
    % Cut out the right portion of the data
    y_now = y(1:T);

    % Memory allocation
    updts = zeros(T,1);
    mu = zeros(model.nx,T);
    S = zeros(model.nx,model.nx,T);
    ESJD = zeros(model.nx,model.nx,T);

    % Fixed initialisation for all methods
    X = Xinit(:,1:T);
    X0 = X0init;

    for(r = 1:par.numMCMC)
        % Select one of the methods to run
        if(sampler == 1)
            [Xtmp,X0] = cpf(y_now, model, par_now, X, X0);
        elseif(sampler == 2)
            [Xtmp,X0] = RLsampler(y_now, model, par_now, X, X0);
        elseif(sampler == 3)
            [Xtmp,X0] = PARsampler(y_now, model, par_now, X, X0);
        else
            error('Unknown sampler');
        end
        
        % Update rate
        updts = updts + all(Xtmp~=X,1)';
        % 1st and 2nd moment
        mu = mu + Xtmp;
        for(d = 1:model.nx); % Outer product for each t / loop over dimension instead of T
            S(:,d,:) = S(:,d,:) + reshape(bsxfun(@times, Xtmp, Xtmp(d,:)),[model.nx,1,T]);
        end
        % ESJD
        diff = Xtmp-X; % [nx,T]
        for(d = 1:model.nx); % Outer product for each t / loop over dimension instead of T
            ESJD(:,d,:) = ESJD(:,d,:) + reshape(bsxfun(@times, diff, diff(d,:)),[model.nx,1,T]);
        end
        X = Xtmp;
        
        % Save intermediate results
        if(r==2 || ~mod(r,100))
            fprintf('Saving id:%i, r:%i\n',id,r);
            savethese(sprintf('%s%s_id%i',pathname,filename,id),id,partable,r,updts,mu,S,ESJD,X,X0);
            fprintf('done!\n');
        end        
    end
end
out = 1;
end