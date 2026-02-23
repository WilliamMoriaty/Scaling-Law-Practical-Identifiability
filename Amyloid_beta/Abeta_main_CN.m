% File: Abeta.m
clc; clear; close all;
rng(42)
% === 1. 加载数据 ===

load('Patient_Data/Patient_751_5.mat');


% === 2. 准备模型参数 (L 矩阵) ===
load('Lp68.mat');
L = Lp;
% === 3. 准备初始条件与参数 ===
% 确保参数是列向量 (136 x 1)
theta = zeros(136, 1);
theta(1:68)   = 0.5 + 0.2 * rand(68, 1);       % Lambda
theta(69:136) = 2.0 + (-0.2 + 0.4 * rand(68, 1)); % K

% 初始状态 y0 (必须是 68x1 列向量)
yzero = Patient_Abeta(1, :)'; 

% 实验数据 yexp (用于拟合)
% Patient_Abeta 通常是 (Time x ROI)，我们需要取第2个时间点以后的数据
yexp3 = Patient_Abeta(2:end, :); 

% 时间跨度
tspan = Patient_Age;

% === 4. 设置优化器 ===
% 注意：'levenberg-marquardt' 不支持 lb/ub 约束。
% 改用 'trust-region-reflective' (默认算法) 以支持边界。
options = optimoptions(@lsqnonlin, ...
    'MaxFunctionEvaluations', 1e6, ...
    'MaxIterations', 20000, ... % 测试时改小一点，实际跑改回 20000
    'Algorithm', 'trust-region-reflective', ... 
    'StepTolerance', 1e-8, ...
    'Display', 'iter'); % 显示迭代过程

% 边界约束
lb = [zeros(68,1); zeros(68,1)]; % Lambda > 0, K > 0
ub = [4*ones(68,1);  6.0*ones(68,1)];

theta0 = theta;

% === 5. 定义目标函数句柄 ===
% 使用匿名函数将固定参数 (L, tspan, yzero, yexp3) 传入
fun = @(theta_opt) Objective_Abeta_Residual(theta_opt, L, tspan, yzero, yexp3);

fprintf('开始优化...\n');

% === 6. 运行 lsqnonlin ===
[theta_new, resnorm, residual, exitflag, output, lambda_out, jacobian] = ...
    lsqnonlin(fun, theta0, lb, ub, options);

fprintf('优化完成。Resnorm: %f\n', resnorm);

% === 7. 简单的绘图验证 ===
% 使用优化后的参数再跑一次模型
sim_res = Objective_Abeta_Residual(theta_new, L, tspan, yzero, yexp3);
% 恢复模拟值 (Residual = Sim - Exp => Sim = Residual + Exp)
% 注意维度展平了，这里只简单画一下残差分布
figure;
plot(residual);
title('拟合残差 (All ROIs & Timepoints)');
xlabel('Data Point Index'); ylabel('Error');
% =========================================================
% === 8. 新增可视化：拟合曲线 vs 实验数据点 (9个指定脑区) ===
% =========================================================
fprintf('正在绘制 ROI 拟合对比图 (9 Subplots)...\n');

% 1. 准备绘图数据 (如果之前已经算过 X_sim，这一步可以跳过)
% 解析优化后的参数
n = size(L, 1);
lambda_opt = theta_new(1:n);
K_opt = theta_new(n+1:end);
save('para_Patient_751.mat',"lambda_opt","K_opt")
[Patient_Age_sim, Patient_sim] = ode15s(@(t,x) abeta_ode_func(t, x, L, lambda_opt, K_opt), Patient_Age, yzero);

% 生成平滑时间点用于画线
t_smooth = linspace(min(tspan), max(tspan), 100);

% 使用优化后的参数解 ODE
[t_sim, X_sim] = ode15s(@(t,x) abeta_ode_func(t, x, L, lambda_opt, K_opt), t_smooth, yzero);

% 2. 绘图设置
% 指定的 9 个脑区
target_rois = [2, 6, 17, 22, 33, 34, 44, 57, 68]; 
num_plots = length(target_rois);

% 调整 Figure 大小以容纳 3x3 网格
figure('Position', [50, 50, 1200, 1000], 'Name', '拟合结果对比: 9个指定脑区');

for k = 1:num_plots
    roi_idx = target_rois(k); % 当前脑区索引
    
    % 使用 3行3列 的子图布局
    subplot(3, 3, k);
    
    % --- A. 画拟合曲线 (ODE 结果 - 蓝色实线) ---
    plot(t_sim, X_sim(:, roi_idx), 'b-', 'LineWidth', 2); 
    hold on;
    
    % --- B. 画实验数据点 (红色圆点) ---
    plot(tspan, Patient_Abeta(:, roi_idx), 'ro', ...
        'MarkerSize', 6, 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
    
    % --- C. 格式美化 ---
    title(['ROI ', num2str(roi_idx)], 'FontSize', 11, 'FontWeight', 'bold');
    
    % 只在最左侧和最下侧的图显示轴标签，保持整洁 (可选)
    if k > 6
        xlabel('Age (Years)');
    end
    if mod(k, 3) == 1
        ylabel('Abeta Load');
    end
    
    % 动态设置 Y 轴范围
    max_val = max([max(X_sim(:, roi_idx)), max(Patient_Abeta(:, roi_idx))]);
    if max_val == 0
        max_val = 1; % 防止全0数据导致坐标轴错误
    end
    ylim([0.2, max_val * 1.2]); 
    xlim([min(tspan), max(tspan)]);
    
    grid on;
    set(gca, 'FontSize', 9);
    hold off;
end

% 添加总标题和图例 (创建一个隐形的坐标轴来放图例)
sgtitle(['Abeta Model Fit vs Data (Optimized) - Resnorm: ' num2str(resnorm)], 'FontSize', 14);

% 在图的底部添加一个共享图例
hL = legend('Fitted ODE', 'Exp Data', 'Orientation', 'horizontal');
set(hL, 'Position', [0.5 - 0.1, 0.02, 0.2, 0.03]); % 手动调整图例位置到最下方


% =========================================================
%           子函数定义 (必须放在文件末尾)
% =========================================================

function residuals = Objective_Abeta_Residual(theta, L, tspan, x0, yexp)
    % lsqnonlin 需要返回“残差向量”，而不是标量 MSE
    
    n = size(L, 1);
    lambda = theta(1:n);
    K = theta(n+1:end);
    
    % 1. 解 ODE (使用 ode15s 处理刚性)
    % 积分区间覆盖 tspan 的范围
    t_integration = [min(tspan), max(tspan)];
    % 防止 tspan 只有一个点导致积分范围出错
    if t_integration(1) == t_integration(2)
        t_integration = [0, tspan(end)];
    end
    
    sol = ode15s(@(t,x) abeta_ode_func(t, x, L, lambda, K), t_integration, x0);
    
    % 2. 提取对应时间点的值 (deval)
    % tspan(2:end) 对应 yexp (因为 yexp 只有第2个点以后的数据)
    target_times = tspan(2:end); 
    solpts = deval(sol, target_times); % 结果是 (ROI x Time)
    
    % 3. 数据对齐
    % yexp 输入通常是 (Time x ROI)，需要转置为 (ROI x Time) 以便相减
    if size(yexp, 1) == length(target_times)
        yexp_aligned = yexp';
    else
        yexp_aligned = yexp;
    end
    
    % 4. 计算残差向量 (向量化)
    % residuals = Simulation - Experiment
    % lsqnonlin 会自动计算 sum(residuals.^2)
    diff_mat = solpts - yexp_aligned;
    residuals = diff_mat(:); % 展平为列向量
    
    % 可以在这里处理 NaN (防止优化器崩溃)
    residuals(isnan(residuals)) = 1e6;
end

function dxdt = abeta_ode_func(t, x, L, lambda, K)
    % 核心 ODE 方程
    term_diffusion = -L * x;
    term_reaction  = lambda .* x .* (K - x);
    dxdt = term_diffusion + term_reaction;
end