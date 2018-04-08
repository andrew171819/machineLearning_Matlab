function gk = gaussianKernel(x1, x2, sigma)
x1 = x1(:);
x2 = x2(:);
gk = 0;
gk = exp(-sum((x1 - x2) .^ 2) / (2 * sigma .^ 2));
end