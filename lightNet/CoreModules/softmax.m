function Y =softmax(X,dzdY)

if(length(size(X))>2)
    %cnn
    E = exp(bsxfun(@minus, X, max(X,[],3))) ;    
    L = sum(E,3) ;
end
if(length(size(X))<=2)
    %mlp
    E = exp(bsxfun(@minus, X, max(X,[],1))) ;
    L = sum(E,1) ;
end

Y = bsxfun(@rdivide, E, L) ;

if nargin <= 1, return ; end

% backward
if(length(size(X))>2)
    Y = Y .* bsxfun(@minus, dzdY, sum(dzdY .* Y, 3)) ;
end
if(length(size(X))<=2)
    Y = Y .* bsxfun(@minus, dzdY, sum(dzdY .* Y, 1)) ;
end