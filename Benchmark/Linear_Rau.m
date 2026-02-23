%% 初始化设置
clear; clc;
N_half = 26;     % 半段点数
scan_delta = 1.0; % 参数扫描的偏移范围
bound_delta = 1.0; % 优化时的局部边界宽度（基于当前 x0 的偏移）

t = [1 2 3 4];
y = [-3 -5 -5 -3];

% 优化配置
options = optimoptions(@lsqnonlin, 'MaxFunctionEvaluations', 10000, 'MaxIterations', 40000, ...
    'Algorithm', 'levenberg-marquardt', 'StepTolerance', 1e-8, 'Display', 'off');

% 参数基准值 [theta1, theta2, theta3, theta4]
theta_base = [2, 0, 0, 0];

results_x = cell(4,1);
results_l = cell(4,1);

%% 循环执行 4 个参数的扫描
for param_idx = 1:4
    % 准备该参数的扫描范围
    center_val = theta_base(param_idx);
    x_right = linspace(center_val, center_val + scan_delta, N_half);
    x_left  = linspace(center_val, center_val - scan_delta, N_half);
    
    l_right = zeros(N_half, 1);
    l_left  = zeros(N_half, 1);
    
    % 确定待优化的参数索引
    opt_indices = setdiff(1:4, param_idx);
    x0_base = theta_base(opt_indices);
    
    % --- 向右扫描 (从中心向右) ---
    x0 = x0_base;
    for i = 1:N_half
        % 动态更新边界：基于当前最优解 x0
        lb = x0 - bound_delta;
        ub = x0 + bound_delta;
        
        fun = @(opt_vars) wrapper_loss(param_idx, x_right(i), opt_vars, t, y);
        x_opt = lsqnonlin(fun, x0, lb, ub, options);
        
        x0 = x_opt; % 热启动更新
        res = wrapper_loss(param_idx, x_right(i), x_opt, t, y);
        l_right(i) = sum(res.^2);
    end
    
    % --- 向左扫描 (从中心向左) ---
    x0 = x0_base;
    for i = 1:N_half
        % 动态更新边界：基于当前最优解 x0
        lb = x0 - bound_delta;
        ub = x0 + bound_delta;
        
        fun = @(opt_vars) wrapper_loss(param_idx, x_left(i), opt_vars, t, y);
        x_opt = lsqnonlin(fun, x0, lb, ub, options);
        
        x0 = x_opt; % 热启动更新
        res = wrapper_loss(param_idx, x_left(i), x_opt, t, y);
        l_left(i) = sum(res.^2);
    end
    
    % 合并数据：翻转左侧并拼接右侧
    results_x{param_idx} = [flipud(x_left(2:end)'); x_right'] - center_val;
    results_l{param_idx} = [flipud(l_left(2:end)); l_right];
end

%% 绘图
figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [222,380,606,491]);
titles = {'\Delta\theta_1', '\Delta\theta_2', '\Delta\theta_3', '\Delta\theta_4'};

    i=1;
    subplot(2, 2, i)
    plot(results_x{i}, results_l{i}, 'k-', 'LineWidth', 1.5);
    hold on;
    % 标记中心点（即原始 theta_base 对应的 Loss）
%     plot(0, results_l{i}(N_half), 'ro', 'MarkerFaceColor', 'r'); 
%     ylim([-2e-3,2e-3])
    xlabel('\Delta\theta_1')
    ylabel('$l(\hat{h},\tilde{\theta}|\theta_1)$','Interpreter','latex')
    set(gca,'FontName','Helvetica','FontSize',20,'FontWeight','bold','linewidth',1.5)
    box off
    
    i=2;
    subplot(2, 2, i)
    plot(results_x{i}, results_l{i}, 'k-', 'LineWidth', 1.5);
    hold on;
    % 标记中心点（即原始 theta_base 对应的 Loss）
%     plot(0, results_l{i}(N_half), 'ro', 'MarkerFaceColor', 'r'); 
    ylim([0,0.03])
    xlabel('\Delta\theta_2')
    ylabel('$l(\hat{h},\tilde{\theta}|\theta_2)$','Interpreter','latex')
    set(gca,'FontName','Helvetica','FontSize',20,'FontWeight','bold','linewidth',1.5)
    box off

    i=3;
    subplot(2, 2, i)
    plot(results_x{i}, results_l{i}, 'k-', 'LineWidth', 1.5);
    hold on;
    % 标记中心点（即原始 theta_base 对应的 Loss）
%     plot(0, results_l{i}(N_half), 'ro', 'MarkerFaceColor', 'r'); 
    ylim([-1e-6,1e-6])
    xlabel('\Delta\theta_3')
    ylabel('$l(\hat{h},\tilde{\theta}|\theta_3)$','Interpreter','latex')
    set(gca,'FontName','Helvetica','FontSize',20,'FontWeight','bold','linewidth',1.5)
    box off

    i=4;
    subplot(2, 2, i)
    plot(results_x{i}, results_l{i}, 'k-', 'LineWidth', 1.5);
    hold on;
    % 标记中心点（即原始 theta_base 对应的 Loss）
%     plot(0, results_l{i}(N_half), 'ro', 'MarkerFaceColor', 'r'); 
    ylim([-1e-6,1e-6])
    xlabel('\Delta\theta_4')
    ylabel('$l(\hat{h},\tilde{\theta}|\theta_4)$','Interpreter','latex')
    set(gca,'FontName','Helvetica','FontSize',20,'FontWeight','bold','linewidth',1.5)
    box off

%% 辅助函数
function L_res = wrapper_loss(fixed_idx, fixed_val, opt_vars, t, y)
    full_theta = zeros(1, 4);
    full_theta(fixed_idx) = fixed_val;
    full_theta(setdiff(1:4, fixed_idx)) = opt_vars;
    L_res = Linear_loss(t, full_theta(1), full_theta(2), full_theta(3), full_theta(4), y);
end

function L_vec = Linear_loss(t, theta1, theta2, theta3, theta4, y)
    H = Hessian_matrix();
    L_vec = zeros(size(t));
    for i = 1:length(t)
        poly_part = (t(i)-1)*(t(i)-2)*(t(i)-3)*(t(i)-4);
        
        % 拟合值计算
        y_hat = theta3*(poly_part + 1.0) + ...
                theta2*(poly_part + 1.5) + ...
                theta4*(poly_part + 0.5) + ...
                theta1*(abs(t(i)-2.5)-3);
        
        % Hessian 修正项
        y_hat = y_hat + ...
                0.5*theta2*H(2,2)*theta2*(t(i)-2)*(t(i)-3)*(t(i)-4)/((1-2)*(1-3)*(1-4)) + ...
                0.5*theta4*H(4,4)*theta4*(t(i)-1)*(t(i)-2)*(t(i)-4)/((3-1)*(3-2)*(3-4)) + ...
                0.5*theta3*H(3,3)*theta3*(t(i)-1)*(t(i)-3)*(t(i)-4)/((2-1)*(2-3)*(2-4)) + ...
                0.5*theta1*H(1,1)*theta1*(t(i)-1)*(t(i)-2)*(t(i)-3)/((4-1)*(4-2)*(4-3));
        
        L_vec(i) = y_hat - y(i);
    end
end

% function H = Hessian_matrix()
%     H = diag([0, 0.5, 0, 0]);
%     H(2, 4) = 1; H(4, 2) = 1;
% end