% File: Abeta.m
clc; clear; close all;

% === 1. 加载数据 ===
% load('Patient_Data/Patient_4598_6.mat');
% 
% load('para_Patient_4598.mat');
load('Patient_Data/Patient_751_5.mat');
% 
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
% % second-order
% U10 = U1(:,nnn1+1:136-nnn0);
% H1 = U00'*H0*U0(:,1:nnn0)*pinv(Sigma_ff(1:nnn0,1:nnn0))*U0(:,1:nnn0)'*H0*U00;
% F2 = U10'*H1*U10;
% [U2,Sigma_ff2,~] = svd(F2);

fig1=figure(1);
clf();
set(gcf,"Position",[141,516,522,343])

alphaData = ones(136,136);  % 初始化为全不透明
alphaData(:, nnn0+1:136) = 0.2;  % 设置右半部分透明度为 0.2

imagesc(abs(U0),'AlphaData',alphaData);
% 设置 colormap
cmap = othercolor('BuDRd_12');
colormap(cmap);  % 可以选择其他 colormap 例如 'jet', 'hot', 'cool' 等
clim([0,1.1])
% 添加 colorbar 并设置标签
c = colorbar;
c.Label.String = '|\partial U_i^T\theta/\partial \theta_j|';  % 设置 colorbar 的标签
c.FontSize = 20;  % 调整字体大小
c.Label.FontWeight = 'bold';  % 设置字体加粗
set(gca,'FontWeight','bold','FontSize',18)
set(gca,'XTick',1:19:136,'XTickLabel',1:19:136,...
    'YTick',1:19:136,'YTickLabel',1:19:136)
xlabel('U','FontSize',30,'FontWeight','bold')
ylabel('\theta','FontSize',30,'FontWeight','bold')

title('$\mathcal{O}(1)$','FontSize',30,'FontWeight','bold','Interpreter','latex')

fig1=figure(2);
clf();
set(gcf,"Position",[141,516,522,343])

alphaData1 = ones(136,136);  % 初始化为全不透明
alphaData1(:, nnn1+nnn0+1:136) = 0.2;  % 设置右半部分透明度为 0.2

imagesc(1:136,1:136,abs(U11),'AlphaData',alphaData1);
% 设置 colormap
cmap = othercolor('BuDRd_12');
colormap(cmap);  % 可以选择其他 colormap 例如 'jet', 'hot', 'cool' 等
clim([0,1.1])
% 添加 colorbar 并设置标签
c = colorbar;
c.Label.String = '|\partial U_i^T\theta/\partial \theta_j|';  % 设置 colorbar 的标签
c.FontSize = 20;  % 调整字体大小
c.Label.FontWeight = 'bold';  % 设置字体加粗
set(gca,'FontWeight','bold','FontSize',18)
set(gca,'XTick',1:19:136,'XTickLabel',1:19:136,...
    'YTick',1:19:136,'YTickLabel',1:19:136)
xlabel('U','FontSize',30,'FontWeight','bold')
ylabel('\theta','FontSize',30,'FontWeight','bold')

title('$\mathcal{O}(\varepsilon)$','FontSize',30,'FontWeight','bold','Interpreter','latex')

fig = figure(3);
clf();
set(gcf,'Position',[309,513,600,270])

bar(1:136,diag(Sigma_ff))
hold on
plot([0.5,136.5],[thres thres],'k--','LineWidth',1.5)

ylim([1e-4,1e2])
ylabel('Eigenvalue');
set(gca,'Ytick',[1e-4,1e-2,1e0,1e2],'xtick',1:19:136,'xticklabel',1:19:136,'Yscale','log')
set(gca,'Yscale','log','FontName','Helvetica','FontSize',16,'FontWeight','bold','linewidth',1.2)

box off

fig = figure(4);
clf();
set(gcf,'Position',[309,513,600,270])

bar(1:136-nnn0,diag(Sigma_ff1),'FaceColor',[0.85,0.33,0.10])
hold on
plot([1-0.5,136.5-nnn0],[thres thres],'k--','LineWidth',1.5)
ylim([1e-4,1e2])
ylabel('Eigenvalue');
set(gca,'Ytick',[1e-4,1e-2,1e0,1e2],'xtick',1:7:136-nnn0,'xticklabel',nnn0+1:7:136,'Yscale','log')
set(gca,'Yscale','log','FontName','Helvetica','FontSize',16,'FontWeight','bold','linewidth',1.2)

box off
