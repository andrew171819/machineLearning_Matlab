function [y,mask] = dropout(x,dzdy,opts)


% determine mask
mask = opts.mask ;
scale = cast(1 / (1 - opts.rate), 'like', x) ;

backMode=1;
if isempty(mask)
    backMode=0;
    if isa(x,'gpuArray')
        mask = scale * (gpuArray.rand(size(x), classUnderlying(x)) >= opts.rate) ;
    else
        mask = scale * (rand(size(x), 'like', x) >= opts.rate) ;     
    end
end

if ~backMode
    y = mask .* x ;   
else
    y = mask .* dzdy ;   
end
