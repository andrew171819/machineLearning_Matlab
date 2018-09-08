function [x, x_dot, theta, theta_dot] = cart_pole(action, x, x_dot, theta, theta_dot)
mass_cart = 1.0;
mass_pole = 0.1;
mass_total = mass_cart + mass_pole;
length = 0.5;
poleMass_length = mass_pole * length;
tau = 0.02;

if action > 0,
    force = 10.0;
else
    force = -10.0;
end

temp = (force + poleMass_length * theta_dot * theta_dot * sin(theta)) / mass_total;
thetaacc = (9.8 * sin(theta) - cos(theta) * temp) / (length * (1.3333333 - mass_pole * cos(theta) * cos(theta) / mass_total));
xacc  = temp - poleMass_length * thetaacc * cos(theta) / mass_total;

x = x + tau * x_dot;
x_dot = x_dot + tau * xacc;
theta = theta + tau * theta_dot;
theta_dot = theta_dot + tau * thetaacc;