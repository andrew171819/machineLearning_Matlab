opts.n_epoch = n_epoch;
opts.dataset_name = dataset_name;
opts.network_name = network_name;
opts.use_gpu = use_gpu;

if ~isfield(opts, 'LoadNet')
end
opts.LoadNet = 0;

opts.dataDir =  ['./', opts.dataset_name, '/'];
opts = PrepareDataFunc(opts);
opts.parameters.current_ep = 1;

opts.parameters.learning_method = learning_method;
opts.parameters.selective_sgd = use_selective_sgd;

if(~exist('init_train', 'var'))
    init_train = 0;
end
opts.parameters.init_train = init_train;

if(~exist('selection_reset_freq', 'var'))
    selection_reset_freq = 0;
end

% sgd parameters
if opts.parameters.selective_sgd == 1
    if ~isfield(opts.parameters,'search_iterations')
        opts.parameters.search_iterations = 30; % iterations used to determine the learning rate
    end
    opts.parameters.ssgd_search_freq = ssgd_search_freq; % search every n epoch
    opts.parameters.selection_reset_freq = selection_reset_freq; % reset every n searches
    if ~isfield(opts.parameters,'lrs')
        opts.parameters.lrs = [1, 0.5]; % initialize selection range
        if ~strcmp(func2str(opts.parameters.learning_method), 'sgd')
            opts.parameters.lrs  = opts.parameters.lrs .* 1e-2;
        end
        opts.parameters.lrs = [opts.parameters.lrs, opts.parameters.lrs * 1e-1, opts.parameters.lrs * 1e-2, opts.parameters . lrs * 1e-3];
    end
    opts.parameters.selection_count = 0;
    opts.parameters.selected_lr = [];
end

if opts.parameters.selective_sgd == 0
    opts.parameters.lr = sgd_lr;
end

% sgd parameters
if ~isfield(opts.parameters, 'mom')
    opts.parameters.mom = 0.9;
end

% adam parameters
if strcmp(func2str(opts.parameters.learning_method), 'adam')
    if ~isfield(opts.parameters, 'mom2')
        opts.parameters.mom2 = 0.999;
    end
end

if ~isfield(opts.parameters, 'batch_size')
    opts.parameters.batch_size = 500;
end
if ~isfield(opts.parameters, 'weightDecay')
    opts.parameters.weightDecay = 1e-4;
end

opts = generate_output_filename(opts);

if ~isfield(opts, 'plot')
    opts.plot = 1;
end

if ~isfield(opts, 'LoadResults')
    opts.LoadResults = 0;
end

if ~opts.LoadResults
    TrainingScript();
end