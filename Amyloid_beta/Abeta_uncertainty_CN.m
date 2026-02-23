% File: Abeta.m
clc; clear; close all;

% === 1. 加载数据 ===
load('Patient_Data/Patient_751_5.mat');

load('para_Patient_751.mat');
lambda_vec = lambda_opt;
K_vec = K_opt;

% === 2. 准备模型参数 (L 矩阵) ===
load('Lp68.mat');
L = Lp;

% 初始状态 y0 (必须是 68x1 列向量)
yzero = Patient_Abeta(1, :)'; 
x0 = yzero;
% 实验数据 yexp (用于拟合)
% Patient_Abeta 通常是 (Time x ROI)，我们需要取第2个时间点以后的数据
yexp3 = Patient_Abeta(2:end, :); 

% 时间跨度
tspan = Patient_Age;

[Time, h_val, y_hat, z_hat] = Abeta_sensitivity_identity(tspan, L, lambda_vec, K_vec, x0);

F = 0;
for i=1:5
y1 = squeeze(y_hat(i,:,:));
F = F + y1'*y1;
end

H = 0;
for i=1:5
for j=1:68
H = H + squeeze(z_hat(i,j,:,:));
end
end

%%
thres = 2e-3;
% zero-order
[U0,Sigma_ff,~]=svd(F);
nnn0 = sum(diag(Sigma_ff)>thres);
H0 = H;
% first-order
U00 = U0(:,nnn0+1:136);
F1 = U00'*H*U00;
[U1,Sigma_ff1,~] = svd(F1);
nnn1 = sum(diag(Sigma_ff1)>thres);
U11 = [U0(:,1:nnn0) U0(:,nnn0+1:136)*U1];
%%
tspan1 = linspace(77.1,87,51);
[Time, h_val, y_hat, z_hat] = Abeta_sensitivity_identity(tspan1, L, lambda_vec, K_vec, x0);
Var = zeros(51,68);
for i=1:51
y1 = squeeze(y_hat(i,:,:));
for j = 1:68
Var(i,j) = y1(j,:)*U0(:,nnn0+1:end)*U0(:,nnn0+1:end)'*y1(j,:)';
end
end

Var1 = sum(Var,2);
abeta = sum(h_val,2);
yexp = sum(Patient_Abeta,2);

sigma_th=2e3;
Y_fit=yexp;

CI=1.96*sqrt(sigma_th*Var1);

Var2 = zeros(51,68);
for i=1:51
y1 = squeeze(y_hat(i,:,:));
for j = 1:68
Var2(i,j) = y1(j,:)*U0(:,nnn0+nnn1+1:end)*U0(:,nnn0+nnn1+1:end)'*y1(j,:)';
end
end

Var3 = sum(Var2,2);

sigma_th=5e11;

CI1=1.96*sqrt(sigma_th*Var3);


%%
fig = figure(1);
clf();
set(gcf,"Position",[478,414,408,272])

plot(Patient_Age,Y_fit,'Marker','o','MarkerSize',10, ...
    'MarkerFaceColor','r','MarkerEdgeColor','r','LineStyle','none')
hold on
% 绘制置信区域（拟合曲线上下的置信区间）
fill([tspan1'; flipud(tspan1')], [abeta + CI; flipud(abeta - CI)], 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
hold on
% 绘制置信区域（拟合曲线上下的置信区间）
fill([tspan1'; flipud(tspan1')], [abeta + CI1; flipud(abeta - CI1)], 'g', 'FaceAlpha', 0.6, 'EdgeColor', 'none');
hold on
plot(tspan1,abeta,'LineWidth',2.0,'Color','k');
hold on

xlim([77,87])
ylim([25,35])
set(gca,'Yscale','linear')
xlabel('Age (years)')
ylabel('A\beta')
set(gca,'FontName','Helvetica','FontSize',18,'FontWeight','bold','linewidth',2.0)
lgd = legend('$\mathbf{data}$', ...
             '$\mathcal{O}(1)$', ...
             '$\mathcal{O}(\varepsilon)$', ...
             '$\mathbf{model}$', ...
             'Interpreter', 'latex');

lgd.ItemTokenSize = [15,6];
% lgd.Position = [0.746 0.301 0.139 0.326];
lgd.Location = 'southeast';
lgd.Box = 'off';
lgd.FontSize = 18;
lgd.FontWeight = 'bold';
box off