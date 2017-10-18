function [ y, dzdw,dzdb ] = fast_mlp_layer( I,weight,bias,dzdy )
%FAST_MLP_LAYER Summary of this function goes here
%   Detailed explanation goes here
%I: input_dim x batch_size

dzdw=[];  
dzdb=[];  
if isempty(dzdy)
    %forward mode

    y=weight*I;
    
    if ~isempty(bias)
        if numel(bias)==numel(y)
            y=y+bias;% much faster
        else
            y=bsxfun(@plus,y,bias);
        end
    end
    
else    
    %backward mode
    
    y=weight'*dzdy;    
    if ~isempty(bias)
        dzdb=mean(dzdy,2);%minibatch averaging    
    end    
    dzdy=permute(dzdy,[1,3,2]);
    I=permute(I,[3,1,2]);
    dzdw=bsxfun(@times,dzdy,I);
    dzdw=mean(dzdw,3);
    
end




