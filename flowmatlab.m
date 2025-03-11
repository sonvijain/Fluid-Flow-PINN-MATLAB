clc; clear; close all;

%% Generate Data
L = 1; % Domain size (square domain)
N = 50; % Number of points in each direction
total_points = N*N;

[x, y] = meshgrid(linspace(0, L, N), linspace(0, L, N));
x = x(:); y = y(:);
t = zeros(total_points, 1); 
tuples = [x, y, t];

% Define simple velocity and pressure fields 
u_true = -sin(pi * x) .* cos(pi * y);
v_true = cos(pi * x) .* sin(pi * y);
p_true = -0.25 * (cos(2 * pi * x) + cos(2 * pi * y));

%% Neural Network
layers = [ 
    featureInputLayer(3, 'Name', 'input')
    fullyConnectedLayer(20, 'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
    fullyConnectedLayer(20, 'Name', 'fc2')
    tanhLayer('Name', 'tanh2')
    fullyConnectedLayer(3, 'Name', 'output') % Outputs: u, v, p
];

net = dlnetwork(layers);
nu = 0.01; % Viscosity

%% Training Setup
numEpochs = 5000;
learningRate = 1e-3;

for epoch = 1:numEpochs
    X_dl = dlarray(tuples', 'CB');
    U_true_dl = dlarray([u_true, v_true, p_true]', 'CB');
    
    [loss, grads] = dlfeval(@lossFunction, net, X_dl, U_true_dl, nu);
    net = dlupdate(@(w,g) w - learningRate * g, net, grads);
    
    if mod(epoch, 500) == 0
        fprintf('Epoch %d - Loss: %.6f\n', epoch, extractdata(loss));
    end
end

%% Prediction and Visualization
U_pred_dl = predict(net, X_dl);
U_pred = extractdata(U_pred_dl)';
u_pred = U_pred(:,1); v_pred = U_pred(:,2); p_pred = U_pred(:,3);

[X, Y] = meshgrid(linspace(0, L, N), linspace(0, L, N));

u_pred = reshape(u_pred, [N, N]);
v_pred = reshape(v_pred, [N, N]);
p_pred = reshape(p_pred, [N, N]);

% Define the analytical solution for Lid-Driven Cavity Flow (Ghia et al. 1982)
[Xg, Yg] = meshgrid(linspace(0, L, N), linspace(0, L, N));
U_analytical = zeros(size(Xg));
V_analytical = zeros(size(Yg));


for i = 1:N
    for j = 1:N
        U_analytical(i, j) = sin(pi * Xg(i, j)) * (1 - cos(pi * Yg(i, j)));
        V_analytical(i, j) = -sin(pi * Yg(i, j)) * (1 - cos(pi * Xg(i, j)));
    end
end

% Enforce boundary conditions
U_analytical(end, :) = 1;  % Top lid moves with velocity U=1
V_analytical(end, :) = 0;  % No penetration at the lid

% Plot 
subplot(1,2,1);
quiver(X, Y, u_pred, v_pred, 'k'); hold on;
streamslice(X, Y, u_pred, v_pred);
colormap('jet');
title('PINN Predicted Streamlines');
xlabel('X'); ylabel('Y');
axis equal; grid on;

subplot(1,2,2);
quiver(Xg, Yg, U_analytical, V_analytical, 'r'); hold on;
streamslice(Xg, Yg, U_analytical, V_analytical);
colormap('jet');
title('Analytical Lid-Driven Cavity Flow');
xlabel('X'); ylabel('Y');
axis equal; grid on;

sgtitle('Comparison of PINN Predicted and Analytical Flow');


function [loss, grads] = lossFunction(net, X, U_true, nu)
    U_pred = forward(net, X);
    
    u = U_pred(1,:); v = U_pred(2,:); p = U_pred(3,:);
    
    du_dx = dlgradient(sum(u), X(1,:), 'EnableHigherDerivatives', true);
    du_dy = dlgradient(sum(u), X(2,:), 'EnableHigherDerivatives', true);
    dv_dx = dlgradient(sum(v), X(1,:), 'EnableHigherDerivatives', true);
    dv_dy = dlgradient(sum(v), X(2,:), 'EnableHigherDerivatives', true);
    dp_dx = dlgradient(sum(p), X(1,:), 'EnableHigherDerivatives', true);
    dp_dy = dlgradient(sum(p), X(2,:), 'EnableHigherDerivatives', true);
    
    d2u_dx2 = dlgradient(sum(du_dx), X(1,:), 'EnableHigherDerivatives', true);
    d2u_dy2 = dlgradient(sum(du_dy), X(2,:), 'EnableHigherDerivatives', true);
    d2v_dx2 = dlgradient(sum(dv_dx), X(1,:), 'EnableHigherDerivatives', true);
    d2v_dy2 = dlgradient(sum(dv_dy), X(2,:), 'EnableHigherDerivatives', true);
    
    continuity = du_dx + dv_dy;
    momentum_x = u .* du_dx + v .* du_dy + dp_dx - nu * (d2u_dx2 + d2u_dy2);
    momentum_y = u .* dv_dx + v .* dv_dy + dp_dy - nu * (d2v_dx2 + d2v_dy2);
    
    physics_loss = mean(continuity.^2 + momentum_x.^2 + momentum_y.^2, 'all');
    data_loss = mean((U_pred - U_true).^2, 'all');
    loss = data_loss + physics_loss;
    grads = dlgradient(loss, net.Learnables);
end
