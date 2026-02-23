clc; clear; close all;
rng(0);
% --- Parameters & Initial Conditions ---
% Order: [lambda, d, k, delta, p, c]
theta0 = [0.065, 0.0065, 0.00064, 0.34, 620, 3.0];
y0 = [10, 0, 1e-6]; % T, I, V

data = readmatrix('HIV_patient3.txt');

idx = randperm(size(data, 1), 3);
idx_sorted = sort(idx);
random_rows = data(idx_sorted, :);

data = data(idx_sorted, :);
tspan = data(:,1) + 35;
xexp3 = data(:,2) * 1e3;

% --- Optimization Settings ---
% Log-space optimization is often more stable for biological rates
lb = [0, 0, 0, 0, 0, 3.0];
ub = [10.0, 1.0, 1e-3, 1.0, 1e3, 3.0];

options = optimoptions('lsqnonlin', ...
    'Display', 'iter', ...
    'FunctionTolerance', 1e-8, ...
    'StepTolerance', 1e-8, ...
    'MaxFunctionEvaluations', 1e5);

% Run Optimization
[theta_new, resnorm] = lsqnonlin(@(th) Objective_HIV(th, tspan, y0, xexp3), ...
                                theta0, lb, ub, options);
l2 = resnorm/size(data,1);
yzero = y0;
tspan1 = linspace(0,120,101);
[t, X1] = ode45(@(t,x)HIV_eq(t,x,theta_new), tspan1, yzero);
XX1 = X1;

%%
% === 1. 初始化设置 ===
    tspan2 = [0;data(:,1)+35];  % 时间区间
    n = 3;            % 状态维度 [T, I, V]
    n_theta = 6;      % 参数维度 [lambda, d, k, delta, p, c]
    
    
    % 初始条件
    x0 = yzero';        % 
    y0 = zeros(n, n_theta);   % 一阶灵敏度初始值
    z0 = zeros(n, n_theta, n_theta); % 二阶灵敏度初始值
    
    % 打包初始状态
    S0 = [x0; y0(:); z0(:)];
    
    % === 2. 调用 ODE 求解器 ===
    options = odeset('RelTol', 1e-5, 'AbsTol', 1e-6);
    [Time, S] = ode45(@(t, S) hiv_ode(t, S, theta_new, n, n_theta), tspan2, S0, options);
    
   
    
    % === 3. 解包结果与输出变量计算 (y_hat, z_hat) ===
    % 这里的 h(x) = log10(V) = log10(x3)
    X = S(:, 1:n); 
    T_cell = X(:,1); I_cell = X(:,2); V_virus = X(:,3);
    
    num_steps = length(Time);
    
    % 初始化输出变量的灵敏度
    % y_hat 维度: (Time x 观测变量维数 x 参数维数) -> (Time x 1 x 6)
    y_hat = zeros(num_steps, n_theta); 
    % z_hat 维度: (Time x 1 x 6 x 6) -> 简化为 (Time x 6 x 6)
    z_hat = zeros(num_steps, n_theta, n_theta);
  
    
    for t = 1:num_steps
        % --- A. 准备当前时刻的数据 ---
        xt = X(t, :)'; 
        V_val = xt(3); % x3
        
        % 提取 Y(t) 矩阵 (3 x 6)
        idx_start_y = n + 1;
        idx_end_y = n + n*n_theta;
        Y_flat = S(t, idx_start_y:idx_end_y);
        Yt = reshape(Y_flat, n, n_theta);
        
        % 提取 Z(t) 张量 (3 x 6 x 6)
        idx_start_z = idx_end_y + 1;
        Z_flat = S(t, idx_start_z:end);
        Zt = reshape(Z_flat, n, n_theta, n_theta);
        
        % --- B. 计算 h(x) 的导数 ---
        % h(x) = log10(x3)
        % dh/dx = [0, 0, 1/(x3*ln10)]
        inv_V_ln10 = 1 / (V_val * log(10));
        dh_dx = [0, 0, inv_V_ln10]; 
        
        % d2h/dx2 (Hessian of h)
        % d(inv_V_ln10)/dV = -1/(V^2 * ln10)
        d2h_dx2 = zeros(3,3);
        d2h_dx2(3,3) = -1 / (V_val^2 * log(10));
        
        % --- C. 计算 y_hat (公式 3) ---
        % y_hat = dh/dx * y
        % (1x3) * (3x6) -> 1x6
        y_hat(t, :) = dh_dx * Yt;
        
        % --- D. 计算 z_hat (公式 5) ---
        % z_hat_jp = sum(d2h_lr * y_lj * y_rp) + sum(dh_l * z_ljp)
        
        for j = 1:n_theta
            for p = 1:n_theta
                
                % Term 1: y(:,j)^T * H_h * y(:,p)
                
                term1 = Yt(:, j)' * d2h_dx2 * Yt(:, p);
                
                % Term 2: dh_dx * z(:, j, p)
                
                term2 = dh_dx * Zt(:, j, p);
                
                z_hat(t, j, p) = term1 + term2;
            end
        end
    end

%%
F = y_hat'*y_hat;
N = size(xexp3,1)+1;
H = squeeze(sum(z_hat(1:N, :, :), 1));

[U,Sigma_F,~]=svd(F);

thres = 1e-3;
% zero-order
[U0,Sigma_ff,~]=svd(F);
nnn0 = sum(diag(Sigma_ff)>thres);
H0 = H;
% first-order
U00 = U0(:,nnn0+1:6);
F1 = U00'*H*U00;
[U1,Sigma_ff1,~] = svd(F1);
nnn1 = sum(diag(Sigma_ff1)>thres);
U11 = [U0(:,1:nnn0) U0(:,nnn0+1:6)*U1];
%%
% === 1. 初始化设置 ===
    tspan2 = [0;data(:,1)+35];  % 时间区间
    n = 3;            % 状态维度 [T, I, V]
    n_theta = 6;      % 参数维度 [lambda, d, k, delta, p, c]
    
    
    % 初始条件
    x0 = yzero';        % 
    y0 = zeros(n, n_theta);   % 一阶灵敏度初始值
    z0 = zeros(n, n_theta, n_theta); % 二阶灵敏度初始值
    
    % 打包初始状态
    S0 = [x0; y0(:); z0(:)];
    
    % === 2. 调用 ODE 求解器 ===
    options = odeset('RelTol', 1e-5, 'AbsTol', 1e-6);
    [Time, S] = ode45(@(t, S) hiv_ode(t, S, theta_new, n, n_theta), tspan1, S0, options);
    
   
    
    % === 3. 解包结果与输出变量计算 (y_hat, z_hat) ===
    % 这里的 h(x) = log10(V) = log10(x3)
    X = S(:, 1:n); 
    T_cell = X(:,1); I_cell = X(:,2); V_virus = X(:,3);
    
    num_steps = length(Time);
    
    % 初始化输出变量的灵敏度
    % y_hat 维度: (Time x 观测变量维数 x 参数维数) -> (Time x 1 x 6)
    y_hat = zeros(num_steps, n_theta); 
    % z_hat 维度: (Time x 1 x 6 x 6) -> 简化为 (Time x 6 x 6)
    z_hat = zeros(num_steps, n_theta, n_theta);
  
    
    for t = 1:num_steps
        % --- A. 准备当前时刻的数据 ---
        xt = X(t, :)'; 
        V_val = xt(3); % x3
        
        % 提取 Y(t) 矩阵 (3 x 6)
        idx_start_y = n + 1;
        idx_end_y = n + n*n_theta;
        Y_flat = S(t, idx_start_y:idx_end_y);
        Yt = reshape(Y_flat, n, n_theta);
        
        % 提取 Z(t) 张量 (3 x 6 x 6)
        idx_start_z = idx_end_y + 1;
        Z_flat = S(t, idx_start_z:end);
        Zt = reshape(Z_flat, n, n_theta, n_theta);
        
        % --- B. 计算 h(x) 的导数 ---
        % h(x) = log10(x3)
        % dh/dx = [0, 0, 1/(x3*ln10)]
        inv_V_ln10 = 1 / (V_val * log(10));
        dh_dx = [0, 0, inv_V_ln10]; 
        
        % d2h/dx2 (Hessian of h)
        % d(inv_V_ln10)/dV = -1/(V^2 * ln10)
        d2h_dx2 = zeros(3,3);
        d2h_dx2(3,3) = -1 / (V_val^2 * log(10));
        
        % --- C. 计算 y_hat (公式 3) ---
        % y_hat = dh/dx * y
        % (1x3) * (3x6) -> 1x6
        y_hat(t, :) = dh_dx * Yt;
        
        % --- D. 计算 z_hat (公式 5) ---
        % z_hat_jp = sum(d2h_lr * y_lj * y_rp) + sum(dh_l * z_ljp)
        
        for j = 1:n_theta
            for p = 1:n_theta
                
                % Term 1: y(:,j)^T * H_h * y(:,p)
                
                term1 = Yt(:, j)' * d2h_dx2 * Yt(:, p);
                
                % Term 2: dh_dx * z(:, j, p)
                
                term2 = dh_dx * Zt(:, j, p);
                
                z_hat(t, j, p) = term1 + term2;
            end
        end
    end

%% 
N2 = 101;
Var = zeros(N2,1);
Var1 = zeros(N2,1);
% thresh = 1e-6;
% r = min(find(diag(Sigma_F)<thresh));

for i=1:N2

Var(i) = y_hat(i,:)*U(:,nnn0+1:end)*U(:,nnn0+1:end)'*y_hat(i,:)';
end
sigma_th=5e-3;
Y_fit=log10(2*XX1(:,3)*1e3);

CI=1.96*sqrt(sigma_th*Var);

for i=1:N2

Var1(i) = y_hat(i,:)*U11(:,nnn0+nnn1+1:end)*U11(:,nnn0+nnn1+1:end)'*y_hat(i,:)';
end
sigma_th=2e4;

CI1=1.96*sqrt(sigma_th*Var1);

%%
fig = figure(1);
clf();
set(gcf,"Position",[478,414,408,272])

plot(tspan,log10(xexp3),'Marker','o','MarkerSize',10, ...
    'MarkerFaceColor','r','MarkerEdgeColor','r','LineStyle','none')
hold on
% 绘制置信区域（拟合曲线上下的置信区间）
fill([tspan1'; flipud(tspan1')], [Y_fit + CI; flipud(Y_fit - CI)], 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
hold on
fill([tspan1'; flipud(tspan1')], [Y_fit + CI1; flipud(Y_fit - CI1)], 'g', 'FaceAlpha', 0.6, 'EdgeColor', 'none');
hold on
plot(tspan1,log10(2*XX1(:,3)*1e3),'LineWidth',2.0,'Color','k');
hold on

xlim([0,120])
ylim([2,7])
set(gca,'Yscale','linear')
xlabel('Day post infection (days)')
ylabel('log_{10} HIV (copies/ml)')
set(gca,'FontName','Helvetica','FontSize',15,'FontWeight','bold','linewidth',2.0)
lgd = legend('$\mathbf{data}$', ...
             '$\mathcal{O}(1)$', ...
             '$\mathcal{O}(\varepsilon)$', ...
             '$\mathbf{model}$', ...
             'Interpreter', 'latex');

lgd.ItemTokenSize = [15,6];
% lgd.Position = [0.746 0.301 0.139 0.326];
lgd.Location = 'northeast';
lgd.Box = 'off';
lgd.FontWeight = 'bold';
box off