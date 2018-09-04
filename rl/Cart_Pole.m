% takes an action (0 or 1) and the current values of the four state variables and updates their values by estimating the state tau seconds later
function [x, x_dot, theta, theta_dot] = Cart_Pole(action, x, x_dot, theta, theta_dot)
g = 9.8;
Mass_Cart = 1.0; % mass of the cart is assumed to be 1 kg
Mass_Pole = 0.1; % mass of the pole is assumed to be 0.1 kg
Total_Mass = Mass_Cart + Mass_Pole;
Length = 0.5; % half of the length of the pole
PoleMass_Length = Mass_Pole * Length;
Force_Mag = 10.0;
Tau = 0.02; % time interval for updating the values
Fourthirds = 1.3333333;

if action > 0,
    force = Force_Mag;
else
    force = -Force_Mag;
end

temp = (force + PoleMass_Length * theta_dot * theta_dot * sin(theta)) / Total_Mass;
thetaacc = (g * sin(theta) - cos(theta) * temp) / (Length * (Fourthirds - Mass_Pole * cos(theta) * cos(theta) / Total_Mass));
xacc  = temp - PoleMass_Length * thetaacc * cos(theta) / Total_Mass;

% update the four state variables, using euler's method
x = x + Tau * x_dot;
x_dot = x_dot + Tau * xacc;
theta = theta + Tau * theta_dot;
theta_dot = theta_dot + Tau * thetaacc;