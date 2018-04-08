function lk = linearKernel(x1, x2)
x1 = x1(:);
x2 = x2(:);
lk = x1' * x2;
end