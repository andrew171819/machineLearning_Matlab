function [ y,from ] = maxpool( I, K, S,pad,dzdy,from,opts )

if exist('opts','var')
   if ~isfield(opts,'parameters')||~isfield(opts.parameters,'eps_pool')
      opts.parameters.eps_pool=0.0; 
   end
else
    opts.training=0;
    opts.parameters.eps_pool=0.0;
end

if isempty(dzdy)
    %%forward
    
    if(~isempty(pad))
       I = pad_data(I,pad,[]);       
    end
    
    [Hin,Win,N,B]=size(I);
    
    if length(K)==1
        K(2)=K(1);
    end
    
    if length(S)==1
        S(2)=S(1);
    end
    
    Hout = ceil((Hin-K(1)+1)/S(1));
    Wout = ceil((Win-K(2)+1)/S(2));
    
    
    [slices,idx0]=im2col_ln(I,K,S);
    [y,from]=max(slices,[],1);
    from=double(from);%
    y=reshape(y,Hout,Wout,N,B);
    from=reshape(from,Hout,Wout,N,B);
    
    if opts.training==1&&opts.parameters.eps_pool>0.0
       p = randperm(numel(from),ceil(numel(from)*opts.parameters.eps_pool));%idx
       from(p)=randi(K*K,[length(p),1]);%random sampled
    end

            
    %now we need to deal with the indexes again.
    idx0=reshape(idx0(1,:,:,:)-1,size(from));
    [r,c]=ind2sub([K,K],from);
    from=r+(c-1).*Hin;
    from=from+idx0;
    
    
else
    %backward
    %dzdy=gather(dzdy);
    
    input_size=size(I);
    original_size_r=input_size(1);
    original_size_c=input_size(2);
    
    if(~isempty(pad))
        input_size(1)=input_size(1)+pad(1)+pad(2);
        input_size(2)=input_size(2)+pad(3)+pad(4);
        
    end

    
    y=zeros(input_size,'like',dzdy);
    %%%
    
    if length(K)==1
        K(2)=K(1);
    end
    if length(S)==1
        S(2)=S(1);
    end
    
    if((K(1)<=S(1))&&(K(2)<=S(2)))
        y(from)=dzdy;%this is faster but can be wrong.
    else
        y=accumarray(from(:),dzdy(:),[prod(input_size),1]);    %fast and correct
        y=reshape(y,input_size);
    end
    
         
    if(~isempty(pad))
        y=y(1+pad(1):1+pad(1)+original_size_r-1,1+pad(3):1+pad(3)+original_size_c-1,:,:);
    end
    
end