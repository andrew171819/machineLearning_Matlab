function [ net,opts ] = selective_sgd( net,opts )
%NET_APPLY_SELECTIVE_SGD Summary of this function goes here
%   Detailed explanation goes here


    if (mod(opts.parameters.current_ep,opts.parameters.ssgd_search_freq)==1||opts.parameters.ssgd_search_freq==1)

        if ~isfield(opts.parameters,'selection_count')
            opts.parameters.selection_count=0;
        end

        if ~isfield(opts.parameters,'init_train')
            opts.parameters.init_train=1;
        end
        
        [lr_best]=select_learning_rate(net,opts);
        opts.parameters.lr=lr_best;

        
        
        
        if(opts.parameters.init_train&&opts.parameters.selection_count==0)
           disp(['Initial training.']);
           opts.parameters.initial_lr=lr_best*0.1;
           opts.parameters.lr=opts.parameters.initial_lr;
           opts.parameters.current_ep=0;
           
        else
            
            if ~isfield(opts.parameters,'selected_lr')
                opts.parameters.selected_lr(1)=lr_best;
            else        
                opts.parameters.selected_lr(end+1)=lr_best;
            end
           
           
        end
        
        
        
        opts.parameters.selection_count=opts.parameters.selection_count+1;
        
        disp(['Selected learning rate: ',num2str(opts.parameters.lr)]);

    end
        
end

