function [ y, dzdw,dzdb,opts ] = fast_conv_layer( I,kernel,bias,stride,pad,dzdy,opts )
%FAST_CONV Summary of this function goes here
%   Detailed explanation goes here
%calculate three ffts and iffts
dzdw=[];  
dzdb=[]; 

flip_kernel=0;
if isfield(opts,'use_cudnn')&&opts.use_cudnn==1 %
    
    if isfield(opts,'use_corr')&&opts.use_corr==0
       kernel=flip(flip(kernel,1),2);
       flip_kernel=1;
    end
    
    if isempty(dzdy)    
        y = vl_nnconv(I, kernel, bias,'pad',pad,'stride',stride);        
    else       
        [y,dzdw,dzdb]= vl_nnconv(I, kernel, bias, dzdy, 'pad',pad,'stride',stride);
        dzdw=dzdw./opts.parameters.batch_size;
        dzdb=dzdb./opts.parameters.batch_size;
        if(flip_kernel)
            dzdw=flip(flip(dzdw,1),2);
        end
    end
    return;
end



if ~isfield(opts,'use_corr')||opts.use_corr==1
   kernel=flip(flip(kernel,1),2);%most existing packages use corr instead of conv 
   flip_kernel=1;
end

[i1,i2,in,b]=size(I);    
    
if(~isempty(pad))
    original_size_r=i1;
    original_size_c=i2;
    i1=i1+pad(1)+pad(2);
    i2=i2+pad(3)+pad(4);
end

[k1,k2,in,out]=size(kernel);    
 
if isempty(dzdy)
    %forward mode, compute the 'valid' convolution using fft
    if(~isempty(pad))
       I = pad_data(I,pad,[]);       
    end
    
    tk=zeros(i1,i2,in,out,'like',I);
    tk(1:k1,1:k2,:,:)=kernel;      
    kernel=tk;
    
    opts.layer{opts.current_layer}.fI=fft2(I); %store result
    opts.layer{opts.current_layer}.fk=fft2(kernel); %store result
    
   
    y=zeros(i1,i2,out,b,'like',I);
    
    for o=1:out
        fft_conv=bsxfun(@times,opts.layer{opts.current_layer}.fI,opts.layer{opts.current_layer}.fk(:,:,:,o));
        fft_conv=sum(fft_conv,3);
        y(:,:,o,:)=real(ifft2(fft_conv));     
    end
    
    y = y(k1:end,k2:end,:,:);
    if ~isempty(bias)
        bias_p=permute(bias(:),[4,3,1,2]);% check this
        y=bsxfun(@plus,y,bias_p);
    end
    
    
    %%%%strided convolution
    if(max(stride)>1)
        y=y(1:stride(1):end,1:stride(2):end,:,:);
    end
    
    
    if opts.training~=1
        opts.layer{opts.current_layer}.fI=[];
        opts.layer{opts.current_layer}.fk=[];
    end
        
else
    %%back prop: load the precomputed ffts and proceed with the
    %%computation.
   
    %%calculate the 'valid' correlation+flipping    
 
    [d1,d2,out,b]=size(dzdy);
    
    td=zeros(i1,i2,out,b,'like',dzdy);
    
    td(1:stride(1):d1*stride(1),1:stride(2):d2*stride(2),:,:)=dzdy;
    dzdy=td;
    clear td;
    fdzdy=fft2(dzdy);
    dzdw=zeros(k1,k2,in,out,'like',I);
   
    
    for o=1:out
        
        fft_corr=bsxfun(@times,opts.layer{opts.current_layer}.fI,conj(fdzdy(:,:,o,:)));
        fft_corr=mean(fft_corr,4); %minibatch averaging
        fft_corr=real(ifft2(fft_corr));
        dzdw(:,:,:,o)= fft_corr(1:k1,1:k2,:,:);% requires thorough understanding of fft, and the shifts 
    end    
    
    if(~flip_kernel)
        dzdw=flip(flip(dzdw,1),2);
    end
        
    if ~isempty(bias)
        dzdb=sum(sum(mean(dzdy,4),1),2);   
        %minibatch averaging + patch summing (note this is how much it changes the final loss)
        dzdb=permute(dzdb,[4,3,2,1]);
    end
    
    %%calculate the 'full' correlation   
    y=zeros(i1,i2,in,b,'like',dzdy);%y=dzdx
    fk=permute(opts.layer{opts.current_layer}.fk,[1,2,4,3]);
    
    for i=1:in        
        fft_corr=bsxfun(@times,fdzdy,conj(fk(:,:,:,i)));
        fft_corr=sum(fft_corr,3);
        y(:,:,i,:)=real(ifft2(fft_corr));
    end
    
    %next line is a dirty circular shift, according to matlab fft implementation.
    y=circshift(y,[(k1-1),(k2-1)]); 
               
    if(~isempty(pad))
        y=y(1+pad(1):1+pad(1)+original_size_r-1,1+pad(3):1+pad(3)+original_size_c-1,:,:);
    end

end




