
if opts.LoadNet~=1
    net=NetInit(opts);

else
    load([fullfile(opts.output_dir,opts.output_name)]);
    opts.parameters=parameters;
    opts.results=results;
end

opts.results=[];
opts.results.TrainEpochError=[];
opts.results.TestEpochError=[];
opts.results.TrainEpochLoss=[];
opts.results.TestEpochLoss=[];
opts.RecordStats=1;
opts.results.TrainLoss=[];
opts.results.TrainError=[];

opts.n_batch=floor(opts.n_train/opts.parameters.batch_size);
opts.n_test_batch=floor(opts.n_test/opts.parameters.batch_size);


if(opts.use_gpu)       
    for i=1:length(net)
        net(i)=SwitchProcessor(net(i),'gpu');
    end
else
    for i=1:length(net)
        net(i)=SwitchProcessor(net(i),'cpu');
    end
end

start_ep=opts.parameters.current_ep;
if opts.plot
    figure1=figure;
end
for ep=start_ep:opts.n_epoch
    
    
    [net,opts]=train_net(net,opts);  
    [opts]=test_net(net,opts);
    opts.parameters.current_ep=opts.parameters.current_ep+1;
    disp(['Epoch ',num2str(ep),' testing error rate: ',num2str(opts.results.TestEpochError(end))])
    
    
    if opts.plot
        subplot(1,2,1); plot(opts.results.TrainEpochError);hold on;plot(opts.results.TestEpochError);hold off;title('Error Rate per Epoch')
        subplot(1,2,2); plot(opts.results.TrainEpochLoss);hold on;plot(opts.results.TestEpochLoss);hold off;title('Loss per Epoch')
        drawnow;
    end
    
    parameters=opts.parameters;
    results=opts.results;
    save([fullfile(opts.output_dir2,[opts.output_name2,num2str(ep),'.mat'])],'net','parameters','results');     
end

