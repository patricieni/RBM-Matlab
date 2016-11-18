clear; close all; clc

%% ================ Part 1: Create Bars-As-Stripes input ================
% Each line in matrix V is a sample: 16 elements long (4x4)
% Generating input
fprintf('Generating input for RBM ...\n')

I = [];

for i=0:15
    vmat = repmat(sscanf(dec2bin(i,4),'%1d')',[4,1]);
    I = [I; reshape(vmat,[16,1])'];
    I = [I; reshape(vmat',[16,1])'];
end

%disp(I)
%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ================ Part 2: Initialize parameters of RBM ================
% 
eps = 0.1;                                  % learning rate
minerr = 0.000000001;                       % converged threshold

timesteps = 4000;
err = 0;                                    % initial error
errvec = zeros(1,timesteps);                % error history
W = (2*rand(16,16)-1)*4*sqrt(3.0/16.0);   
%disp(W)                                     % initial weights
v_bias = zeros(1,16);                        % initial bias in visible
h_bias = zeros(1,16);                        % initial bias in hidden
m = size(I,1);

%% ================ Part 3: Train the RBM ================
% We are using 1-step constrastive divergence

for i = 1:timesteps
    
    ordering = randperm(m);
    I = I(ordering, :);
    
    %for j = 1:m
        % Find hidden units by sampling
        hidden_p = sigmoid(I * W + repmat(h_bias,[m,1]));
        H = zeros(size(hidden_p));                           
        H(hidden_p >= rand(size(hidden_p))) = 1;                  
        
        % Find visible units by sampling from the hidden ones.
        visible_p = sigmoid(H * W' +repmat(v_bias,[m,1]));
        V = zeros(size(visible_p));
        V(visible_p >= rand(size(visible_p))) = 1;

        % Last step to find the last kth
        bP = sigmoid(V * W + repmat(h_bias,[m,1]));
        last_H = zeros(size(bP));
        last_H(bP >= rand(size(bP))) = 1;
    
        % Positive Divergence
        % <v_i * p(h_i)>
        pD = I'*hidden_p;
    
        % Negative Divergence
        % <v_k * p(h_k)>
        nD = V'*bP;

        % Find the weights using contrastive divergence
        W = W + eps*(pD - nD)/m;

        % Update the biases
        v_bias = v_bias + eps*(sum(I-V));
        h_bias = h_bias + eps*(sum(hidden_p-bP));    

        % Estimate error by checking how much the biases and weights changed
        format long;
        err = sum(sum((I-visible_p).^2));
        %+(nH-H).^2))+sum(sum((nV'*nH-V'*H).^2)))
        errvec(i) = sqrt(err);
end

%  ---------------------------------------------------------

plot(errvec);

fprintf('Program paused. Press enter to continue.\n');
pause;
%% ================ Part 4: Reconstruct partial data ================
% Use Gibbs Sampling for n steps

%  ----------------- test the machine ----------------------

v1 = [1,0,1,0;0.5,0.5,0.5,0.5;0.5,0.5,0.5,0.5;0.5,0.5,0.5,0.5;];
test_v1 = [1,0,1,0;1,0,1,0;1,0,1,0;1,0,1,0];
test_v1 = reshape(test_v1,[1,16]);
v1 = reshape(v1,[1,16]);
all_v = zeros(10000,1);
err_v1 = 0;

for i = 1:10000
    v = gibbs(v1,h_bias,v_bias,W,1);
    
    all_v(i) = bin2dec(num2str(v));
    
    e = sqrt(sum(sum((v-test_v1).^2)));
    if (e == 0)
        err_v1 = err_v1+ 1;
    end
end
total = err_v1/10000.0;
fprintf('%f \n',total);
hist(all_v,10000);

%  ---------------------------------------------------------
% reshape(v,[4,4]);
fprintf('Program finished. Press enter to exit.\n');
pause;