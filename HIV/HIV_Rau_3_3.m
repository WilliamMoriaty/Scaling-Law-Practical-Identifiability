clc; clear; close all;
rng(0);
% --- Parameters & Initial Conditions ---
% Order: [lambda, d, k, delta, p, c]

% --- 1. 初始化与原始拟合 ---
theta_best = [0.065, 0.0065, 0.00064, 0.34, 620, 3.0];
y0 = [10, 0, 1e-6]; % T, I, V

data = readmatrix('HIV_patient3.txt');

idx = randperm(size(data, 1), 3);
idx_sorted = sort(idx);
random_rows = data(idx_sorted, :);

data = data(idx_sorted, :);
tspan = data(:,1) + 35;
xexp3 = data(:,2) * 1e3;
options = optimoptions('lsqnonlin', 'Display', 'off', 'FunctionTolerance', 1e-8, 'StepTolerance', 1e-8);

% 第一次全局优化确定基准
lb_global = [0, 0, 0, 0, 0, 3.0];
ub_global = [10.0, 1.0, 1e-3, 1.0, 1e3, 3.0];
[theta_opt, resnorm_opt] = lsqnonlin(@(th) Objective_HIV(th, tspan, y0, xexp3), theta_best, lb_global, ub_global, options);

lb = [0, 0, 0, 0, 0, 3.0];
ub = [10.0, 1.0, 1e-3, 1.0, 1e3, 3.0];


% --- 2. 参数敏感性分析 (Profile Likelihood) ---
N = 51;
lam = 0.05; % 
delta_pi = 150;
delta_c = 1.0;

% 预分配空间
pi_range_pos = linspace(theta_opt(5), theta_opt(5) + 10*delta_pi, N);
pi_range_neg = linspace(theta_opt(5) - 3*delta_pi, theta_opt(5), N);
c_range_pos  = linspace(theta_opt(6), theta_opt(6) + 3*delta_c, N);
c_range_neg  = linspace(theta_opt(6) - 1.5*delta_c, theta_opt(6), N);

l_pi = zeros(2*N, 1); pi_vals = [pi_range_neg, pi_range_pos];
l_c  = zeros(2*N, 1); c_vals  = [c_range_neg, c_range_pos];

% --- 分析参数 pi (theta(5)) ---
fprintf('Analyzing parameter: pi...\n');
for i = 1:2*N
    p_fixed = pi_vals(i);
    % 待优化参数: [lambda, d, k, delta, c] (排除 pi)
    x0_sub = [theta_opt(1:4), theta_opt(6)];
    lb_sub = x0_sub * (1 - lam); ub_sub = x0_sub * (1 + lam);
    lb_sub(3) = (1-5*lam)*x0_sub(3); ub_sub(3) = (1+5*lam)*x0_sub(3); 
    [theta_sub, ~] = lsqnonlin(@(x) Objective_HIV_fixed(x, p_fixed, 5, tspan, y0, xexp3), x0_sub, lb_sub, ub_sub, options);
    full_theta = [theta_sub(1:4), p_fixed, theta_sub(5)];
    l_pi(i) = Calculate_Loss(full_theta, tspan, y0, xexp3);
end
lam = 0.01;

% --- 分析参数 c (theta(6)) ---
fprintf('Analyzing parameter: c...\n');
for i = 1:2*N
    c_fixed = c_vals(i);
    % 待优化参数: [lambda, d, k, delta, pi] (排除 c)
    x0_sub = theta_opt(1:5);
    lb_sub = x0_sub * (1 - lam); ub_sub = x0_sub * (1 + lam);
    [theta_sub, ~] = lsqnonlin(@(x) Objective_HIV_fixed(x, c_fixed, 6, tspan, y0, xexp3), x0_sub, lb_sub, ub_sub, options);
    full_theta = [theta_sub, c_fixed];
    l_c(i) = Calculate_Loss(full_theta, tspan, y0, xexp3);
end

% --- 3. 绘图 ---

fig1=figure(1);
clf();
set(gcf,'Position',[623,424,358,243])
plot((pi_vals - theta_opt(5))/theta_opt(5), l_pi, 'b-', 'LineWidth', 2, 'DisplayName', '\pi');
hold on;
plot((c_vals - theta_opt(6))/theta_opt(6), l_c, 'r-', 'LineWidth', 2, 'DisplayName', 'c');
xlim([-0.3 0.5])
ylim([-0.05,2.0]) 

xlabel('\Delta\theta_i / \theta_{opt}');
ylabel('$l(\theta|\theta_i)$', 'Interpreter', 'latex');
set(gca,'FontName','Helvetica','FontSize',15,'FontWeight','bold','linewidth',1.2)
box off;
lgd=legend('\pi','c');
lgd.Box='off';
lgd.Location = 'northeast';
lgd.FontSize = 15;
lgd.ItemTokenSize = [10,6];
box off

% function resid = Objective_HIV(theta, tspan, x0, yexp)
%     sol = ode45(@(t,x) HIV_eq(t,x,theta), [0; max(tspan)], x0);
%     V_model = deval(sol, tspan, 3)' * 2000;
%     resid = log10(V_model + eps) - log10(yexp + eps);
% end

function resid = Objective_HIV_fixed(sub_theta, fixed_val, fixed_idx, tspan, x0, yexp)
    % 动态组装完整的 theta 向量
    if fixed_idx == 5
        full_theta = [sub_theta(1:4), fixed_val, sub_theta(5)];
    elseif fixed_idx == 6
        full_theta = [sub_theta(1:5), fixed_val];
    end
    resid = Objective_HIV(full_theta, tspan, x0, yexp);
end

function loss = Calculate_Loss(theta, tspan, x0, yexp)
    % 统一的 Loss 计算公式: MSE in log space
    resid = Objective_HIV(theta, tspan, x0, yexp);
    loss = 0.5 * sum(resid.^2) / length(yexp);
end

% function dxdt = HIV_eq(t, x, theta)
%     % theta = [lambda, d, k, delta, p, c]
%     dxdt = [
%         theta(1) - theta(2)*x(1) - theta(3)*x(3)*x(1);
%         theta(3)*x(3)*x(1) - theta(4)*x(2);
%         theta(5)*x(2) - theta(6)*x(3)
%     ];
% end