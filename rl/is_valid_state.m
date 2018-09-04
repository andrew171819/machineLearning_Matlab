function valid = is_valid_state(x, x_dot, theta, theta_dot)
twelve_degrees = 0.2094384;
valid = 1;
if (x < -2.4 | x > 2.4  | theta < -twelve_degrees | theta > twelve_degrees)
    valid = -1;
end