function [  net,res,opts ] = rmsprop(  net,res,opts )
%NET_APPLY_GRAD_SGD Summary of this function goes here
%   Detailed explanation goes here

    if ~isfield(opts.parameters,'weightDecay')
        opts.parameters.weightDecay=1e-4;
    end
    
    if ~isfield(opts.results,'lrs')
        opts.results.lrs=[];%%not really necessary
    end
    opts.results.lrs=[opts.results.lrs;gather(opts.parameters.lr)];
    
    if ~isfield(opts.parameters,'eps')
        opts.parameters.eps=1e-6;
    end
    
    if ~isfield(opts.parameters,'clip')
        opts.parameters.clip=1e0;
    end
    
    if ~isfield(net,'iterations')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
        net.iterations=0;
    end
    
    net.iterations=net.iterations+1;
    
    mom_factor=(1-opts.parameters.mom.^net.iterations);
    
    for layer=1:numel(net.layers)
        if isfield(net.layers{1,layer},'weights')
            if ~isfield(net.layers{1,layer},'momentum')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
                net.layers{1,layer}.momentum{1}=zeros(size(net.layers{1,layer}.weights{1}),'like',net.layers{1,layer}.weights{1});
                net.layers{1,layer}.momentum{2}=zeros(size(net.layers{1,layer}.weights{2}),'like',net.layers{1,layer}.weights{2});
                opts.reset_mom=0;
            end
            
            net.layers{1,layer}.momentum{1}=opts.parameters.mom.*net.layers{1,layer}.momentum{1}+(1-opts.parameters.mom).*res(layer).dzdw.^2;
            
            normalized_grad=res(layer).dzdw./(net.layers{1,layer}.momentum{1}.^0.5+opts.parameters.eps)./mom_factor;
            if isfield(opts.parameters,'clip')
                mask=abs(normalized_grad)>opts.parameters.clip;
                normalized_grad(mask)=sign(normalized_grad(mask)).*opts.parameters.clip;
            end
            net.layers{1,layer}.weights{1}=net.layers{1,layer}.weights{1}-opts.parameters.lr*normalized_grad- opts.parameters.weightDecay * net.layers{1,layer}.weights{1};
            
            net.layers{1,layer}.momentum{2}=opts.parameters.mom.*net.layers{1,layer}.momentum{2}+(1-opts.parameters.mom).*res(layer).dzdb.^2;
            normalized_grad=res(layer).dzdb./(net.layers{1,layer}.momentum{2}.^0.5+opts.parameters.eps)./mom_factor;
            if isfield(opts.parameters,'clip')
                mask=abs(normalized_grad)>opts.parameters.clip;
                normalized_grad(mask)=sign(normalized_grad(mask)).*opts.parameters.clip;
            end
            net.layers{1,layer}.weights{2}=net.layers{1,layer}.weights{2}-opts.parameters.lr*normalized_grad;
        end
    end
   
end

