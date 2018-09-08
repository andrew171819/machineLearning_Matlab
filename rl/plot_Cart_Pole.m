function plot_cart_pole(x, theta)

pxg = [x + 1 x - 1 x - 1 x + 1 x + 1];
pyg = [0.25 0.25 1.25 1.25 0.25];

pxp = [x x + 2 * sin(theta)];
pyp = [1.25 1.25 + 2 * cos(theta)];

[pxw1, pyw1] = plotcircle(x - 0.5, 0.125, 0, 0.125);
[pxw2, pyw2] = plotcircle(x + 0.5, 0.125, 0, 0.125);

plot(pxg, pyg, 'k-', pxw1, pyw1, 'k', pxw2, pyw2, 'k', pxp, pyp, 'r');
axis([-6 6 0 6]);

grid;
drawnow;