function [ net,y,dzdw,dzdb,opts ] = bnorm( net,x,layer_idx,dzdy,opts )
%BNORM Summary of this function goes here
%   Detailed explanation goes here
    dzdw=[];
    dzdb=[];
    if ~isfield(net,'iterations')
        net.iterations=0;
    end
    if ~isfield(net,'iterations_bn')
        net.iterations_bn=0; 
    elseif isfield(net,'frame_iterations')
        net.iterations_bn=net.frame_iterations;
    else
        net.iterations_bn=net.iterations;
    end
    
    
    if ~isfield(net.layers{1,layer_idx},'weights')        
        sz=size(x);
        sz=[sz(1:end-1),1];
        if length(sz)==4
            sz(1:2)=1;
        end        
        net.layers{1,layer_idx}.weights{1}=ones(sz,'like',x);
        for i=2:4
            net.layers{1,layer_idx}.weights{i}=zeros(sz,'like',x);
        end
        for i=1:2
            net.layers{1,layer_idx}.momentum{i}=zeros(sz,'like',x);
        end        
    end

    if ~isfield(opts.parameters, 'eps_bn')
        opts.parameters.eps_bn=1e-2;
    end
    
   

    if ~isfield(opts.parameters, 'mom_bn')
        opts.parameters.mom_bn=0.01;
    end
        
    if ~isfield(opts.parameters, 'simple_bn')
        opts.parameters.simple_bn=1;
    end
    mom_factor=1-opts.parameters.mom_bn.^(net.iterations_bn+1);
    
    if(length(size(x))>2)
        batch_dim=4; 
    else
        batch_dim=2;
    end
    
    if(opts.training&&isempty(dzdy))
        if batch_dim==4    
            net.layers{1,layer_idx}.weights{3}=opts.parameters.mom_bn*net.layers{1,layer_idx}.weights{3}+(1-opts.parameters.mom_bn)*mean(mean(mean(x,batch_dim),1),2);
            net.layers{1,layer_idx}.weights{4}=opts.parameters.mom_bn*net.layers{1,layer_idx}.weights{4}+(1-opts.parameters.mom_bn)*mean(mean(mean(abs(x).^2,batch_dim),1),2);
        else
            net.layers{1,layer_idx}.weights{3}=opts.parameters.mom_bn*net.layers{1,layer_idx}.weights{3}+(1-opts.parameters.mom_bn)*mean(x,batch_dim);   
            net.layers{1,layer_idx}.weights{4}=opts.parameters.mom_bn*net.layers{1,layer_idx}.weights{4}+(1-opts.parameters.mom_bn)*mean(abs(x).^2,batch_dim);  
        end
        
    end
            
    if(isempty(dzdy))
        
        net.layer{opts.current_layer}.x_n=bsxfun(@minus,x,net.layers{1,layer_idx}.weights{3}./mom_factor);
        net.layer{opts.current_layer}.x_n=bsxfun(@rdivide,net.layer{opts.current_layer}.x_n,(net.layers{1,layer_idx}.weights{4}./mom_factor+opts.parameters.eps_bn).^0.5);
        
        y=bsxfun(@times,net.layer{opts.current_layer}.x_n,net.layers{1,layer_idx}.weights{1});
        y=bsxfun(@plus,y,net.layers{1,layer_idx}.weights{2});
        
    else
        dzdw=mean(dzdy.*net.layer{opts.current_layer}.x_n,batch_dim);
        dzdb=mean(dzdy,batch_dim);
        
        denom=1;
        if batch_dim==4
            denom=size(dzdw,1)*size(dzdw,2);
            dzdw=sum(sum(dzdw,1),2);
            dzdb=sum(sum(dzdb,1),2);
            
        end
        
        if ~opts.parameters.simple_bn
            %the complicated version         
            tmp=bsxfun(@minus,x,net.layers{1,layer_idx}.weights{3}./mom_factor);
            tmp=bsxfun(@rdivide,tmp,(net.layers{1,layer_idx}.weights{4}./mom_factor+opts.parameters.eps_bn).^(0.5));
            tmp=bsxfun(@times,dzdw./denom,tmp);
            tmp=bsxfun(@plus,dzdb./denom,tmp);
            dzdy=(dzdy-tmp.*(1-opts.parameters.mom_bn));

            %[max(abs(dzdy(:))), max(abs(tmp(:)))]
        end
        
        %the simple version:
        y=bsxfun(@times,dzdy,net.layers{1,layer_idx}.weights{1});
        y=bsxfun(@rdivide,y,(net.layers{1,layer_idx}.weights{4}./mom_factor+opts.parameters.eps_bn).^(0.5));

    end
        
end
