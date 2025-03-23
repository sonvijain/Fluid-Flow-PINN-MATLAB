clc; clear; close all;

%% Parameters
L = 1; % Domain size (square domain)
N = 50; % Number of points in each direction
nu = 0.01; % Viscosity
numEpochs = 5000;
learningRate = 1e-3;

%% Generate Data
[x, y] = meshgrid(linspace(0, L, N), linspace(0, L, N));
x = x(:); y = y(:);
t = zeros(size(x));
X_data = [x, y, t]';

% Define synthetic velocity and pressure fields
u_true = sin(pi * x) .* (1 - cos(pi * y));
v_true = -sin(pi * y) .* (1 - cos(pi * x));
p_true = -0.25 * (cos(2 * pi * x) + cos(2 * pi * y));
U_true = [u_true, v_true, p_true]';

%% Define Neural Network
layers = [ 
    featureInputLayer(3, 'Name', 'input')
    fullyConnectedLayer(20, 'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
    fullyConnectedLayer(20, 'Name', 'fc2')
    tanhLayer('Name', 'tanh2')
    fullyConnectedLayer(3, 'Name', 'output')
];
net = dlnetwork(layers);

%% Split data 
numTrain = round(0.8 * size(X_data, 2));
X_train = X_data(:, 1:numTrain);
U_train = U_true(:, 1:numTrain);
X_val = X_data(:, numTrain+1:end);
U_val = U_true(:, numTrain+1:end);

%% Training Loop 
bestLoss = inf;
patience = 500; % Early stopping patience
patienceCounter = 0;

for epoch = 1:numEpochs
    X_dl = dlarray(X_train, 'CB');
    U_true_dl = dlarray(U_train, 'CB');
    
    [loss, grads] = dlfeval(@lossFunction, net, X_dl, U_true_dl, nu);
    net = dlupdate(@(w,g) w - learningRate * g, net, grads);
    
    % Validation loss
    X_val_dl = dlarray(X_val, 'CB');
    U_val_dl = dlarray(U_val, 'CB');
    val_loss = dlfeval(@lossFunction, net, X_val_dl, U_val_dl, nu);
    
    if mod(epoch, 500) == 0
        fprintf('Epoch %d - Loss: %.6f - Val Loss: %.6f\n', epoch, extractdata(loss), extractdata(val_loss));
    end
    
    % Early stopping check
    if val_loss < bestLoss
        bestLoss = val_loss;
        patienceCounter = 0;
    else
        patienceCounter = patienceCounter + 1;
    end
    if patienceCounter >= patience
        fprintf('Early stopping triggered at epoch %d!\n', epoch);
        break;
    end
end

%% Prediction and Visualization
U_pred_dl = predict(net, dlarray(X_data, 'CB'));
U_pred = extractdata(U_pred_dl)';
u_pred = reshape(U_pred(:,1), [N, N]);
v_pred = reshape(U_pred(:,2), [N, N]);
p_pred = reshape(U_pred(:,3), [N, N]);

% Plot results
figure;
quiver(x, y, u_pred(:), v_pred(:));
hold on;
streamslice(x, y, u_pred, v_pred);
colormap('jet');
title('PINN Predicted Streamlines');
xlabel('X'); ylabel('Y');
axis equal; grid on;

%% Loss Function
function [loss, grads] = lossFunction(net, X, U_true, nu)
    U_pred = forward(net, X);
    u = U_pred(1,:); v = U_pred(2,:); p = U_pred(3,:);
    
    % Compute gradients
    du_dx = dlgradient(sum(u), X(1,:), 'EnableHigherDerivatives', true);
    dv_dy = dlgradient(sum(v), X(2,:), 'EnableHigherDerivatives', true);
    dp_dx = dlgradient(sum(p), X(1,:), 'EnableHigherDerivatives', true);
    dp_dy = dlgradient(sum(p), X(2,:), 'EnableHigherDerivatives', true);
    
    d2u_dx2 = dlgradient(sum(du_dx), X(1,:), 'EnableHigherDerivatives', true);
    d2v_dy2 = dlgradient(sum(dv_dy), X(2,:), 'EnableHigherDerivatives', true);
    
    % Physics losses
    continuity = du_dx + dv_dy;
    momentum_x = u .* du_dx + v .* dp_dx - nu * d2u_dx2;
    momentum_y = u .* dp_dy + v .* dv_dy - nu * d2v_dy2;
    
    physics_loss = mean(continuity.^2 + momentum_x.^2 + momentum_y.^2, 'all');
    data_loss = mean((U_pred - U_true).^2, 'all');
    
    % Boundary conditions
    top_lid = X(2,:) == 1;
    boundary_loss = mean((u(top_lid) - 1).^2 + v(top_lid).^2, 'all');
    
    loss = data_loss + physics_loss + boundary_loss;
    grads = dlgradient(loss, net.Learnables);
end
