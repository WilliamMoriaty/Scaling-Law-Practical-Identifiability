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

[U,Sigma_F,~]=svd(F);
nnD = [];
K001 = [];
K010 = [];
K100 = [];
for ND = 1:136
P = eye(136);
P(:,[ND,136]) = P(:,[136,ND]);
P1 = P;
F1 = P1'*F*P1;
H1 = P1'*H*P1;

F11 = F1(1:135,1:135);
[U1,Sigma1,~] = svd(F11);

nn1 = rank(Sigma1);
nnD = [nnD;nn1];
F12 = F1(1:135,136);
F21 = F1(136,1:135);
F22 = F1(136,136);
K01 = F22 - F21 * pinv(F11,1e-2) * F12;

G1 = F21*U1(:,1:nn1);G2 = F21*U1(:,nn1+1:135);
% 
H11 = H1(1:135,1:135);
H12 = H1(1:135,136);
H21 = H1(136,1:135);
H22 = H1(136,136);
% 
 
K1 = H21*U1(:,1:nn1);K2 = H21*U1(:,nn1+1:135);
Sigma = Sigma1(1:nn1,1:nn1);
H11 = U1'*H11*U1;
h11 = H11(1:nn1,1:nn1);
h12 = H11(1:nn1,nn1+1:135);
h22 = H11(nn1+1:135,nn1+1:135);
K11 = H22 - (K1*pinv(Sigma)*G1'+G1*pinv(Sigma)*K1'+K2*pinv(h22)*K2'+G1*pinv(Sigma)*h12*pinv(h22)*h12'*pinv(Sigma)*G1')...
    +G1*pinv(Sigma)*h11*pinv(Sigma)*G1'+K2*pinv(h22)*h12'*pinv(Sigma)*G1'+G1*pinv(Sigma)*h12*pinv(h22)*K2';

term1 = K1*pinv(Sigma)*K1' - G1*pinv(Sigma)*h11*pinv(Sigma)*K1' - K1*pinv(Sigma)*h11*pinv(Sigma)*G1'...
    + G1*(pinv(Sigma)*h11*pinv(Sigma)*h11*pinv(Sigma))*G1';
term2 = (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12'*pinv(Sigma)*h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term3 = (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma)) * (h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term4 = (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12') * (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma))';

K21 = -(term1 + term2 + term3 + term4);

K001=[K001; K01];

K010=[K010; K11];

K100=[K100; K21];
end

fig = figure(1);
clf();
set(gcf,'Position',[309,513,600,270])

% subplot(2,2,1)
SS1=[K001(1:17)';...
    K010(1:17)'];

bar(1:17,SS1)
hold on
ylim([1e-3,1e6])
ylabel('$\mathcal{K}$', 'Interpreter', 'latex');
set(gca,'Ytick',[1e-3,1e0,1e3,1e6],'xtick',1:17,'xticklabel',1:17,'Yscale','log')
set(gca,'Yscale','log','FontName','Helvetica','FontSize',16,'FontWeight','bold','linewidth',1.2)
lgd=legend('$\mathcal{K}_0$','$\mathcal{K}_1$', 'Interpreter', 'latex');
lgd.Box='off';
lgd.FontSize = 15;
lgd.ItemTokenSize = [10,6];
box off

fig = figure(2);
clf();
set(gcf,'Position',[309,513,600,270])

SS2=[K001(18:34)';...
    K010(18:34)'];

bar(1:17,SS2)
hold on
ylim([1e-3,1e6])
ylabel('$\mathcal{K}$', 'Interpreter', 'latex');
set(gca,'Ytick',[1e-3,1e0,1e3,1e6],'xtick',1:17,'xticklabel',18:34,'Yscale','log')
set(gca,'Yscale','log','FontName','Helvetica','FontSize',16,'FontWeight','bold','linewidth',1.2)
lgd=legend('$\mathcal{K}_0$','$\mathcal{K}_1$', 'Interpreter', 'latex');
lgd.Box='off';
lgd.FontSize = 15;
lgd.ItemTokenSize = [10,6];
box off

fig = figure(3);
clf();
set(gcf,'Position',[309,513,600,270])

SS3=[K001(35:51)';...
    K010(35:51)'];

bar(1:17,SS3)
hold on
ylim([1e-3,1e6])
ylabel('$\mathcal{K}$', 'Interpreter', 'latex');
set(gca,'Ytick',[1e-3,1e0,1e3,1e6],'xtick',1:17,'xticklabel',35:51,'Yscale','log')
set(gca,'Yscale','log','FontName','Helvetica','FontSize',16,'FontWeight','bold','linewidth',1.2)
lgd=legend('$\mathcal{K}_0$','$\mathcal{K}_1$', 'Interpreter', 'latex');
lgd.Box='off';
lgd.FontSize = 15;
lgd.ItemTokenSize = [10,6];
box off
% 
fig = figure(4);
clf();
set(gcf,'Position',[309,513,600,270])

SS4=[K001(52:68)';...
    K010(52:68)'];

bar(1:17,SS4)
hold on
ylim([1e-3,1e6])
ylabel('$\mathcal{K}$', 'Interpreter', 'latex');
set(gca,'Ytick',[1e-3,1e0,1e3,1e6],'xtick',1:17,'xticklabel',52:68,'Yscale','log')
set(gca,'Yscale','log','FontName','Helvetica','FontSize',16,'FontWeight','bold','linewidth',1.2)
lgd=legend('$\mathcal{K}_0$','$\mathcal{K}_1$', 'Interpreter', 'latex');
lgd.Box='off';
lgd.FontSize = 15;
lgd.ItemTokenSize = [10,6];
box off

% %
fig = figure(5);
clf();
set(gcf,'Position',[309,513,600,270])

SS5=[K001(69:85)';...
    K010(69:85)';];

bar(1:17,SS5)
hold on
ylim([1e-3,1e6])
ylabel('$\mathcal{K}$', 'Interpreter', 'latex');
set(gca,'Ytick',[1e-3,1e0,1e3,1e6],'xtick',1:17,'xticklabel',1:17,'Yscale','log')
set(gca,'Yscale','log','FontName','Helvetica','FontSize',16,'FontWeight','bold','linewidth',1.2)
lgd=legend('$\mathcal{K}_0$','$\mathcal{K}_1$', 'Interpreter', 'latex');
lgd.Box='off';
lgd.FontSize = 15;
lgd.ItemTokenSize = [10,6];
box off
% 
fig = figure(6);
clf();
set(gcf,'Position',[309,513,600,270])

SS6=[K001(86:102)';...
    K010(86:102)';];

bar(1:17,SS6)
hold on
ylim([1e-3,1e6])
ylabel('$\mathcal{K}$', 'Interpreter', 'latex');
set(gca,'Ytick',[1e-3,1e0,1e3,1e6],'xtick',1:17,'xticklabel',18:34,'Yscale','log')
set(gca,'Yscale','log','FontName','Helvetica','FontSize',16,'FontWeight','bold','linewidth',1.2)
lgd=legend('$\mathcal{K}_0$','$\mathcal{K}_1$', 'Interpreter', 'latex');
lgd.Box='off';
lgd.FontSize = 15;
lgd.ItemTokenSize = [10,6];
box off

fig = figure(7);
clf();
set(gcf,'Position',[309,513,600,270])

SS7=[K001(103:119)';...
    K010(103:119)'];

bar(1:17,SS7)
hold on
ylim([1e-3,1e6])
ylabel('$\mathcal{K}$', 'Interpreter', 'latex');
set(gca,'Ytick',[1e-3,1e0,1e3,1e6],'xtick',1:17,'xticklabel',35:51,'Yscale','log')
set(gca,'Yscale','log','FontName','Helvetica','FontSize',16,'FontWeight','bold','linewidth',1.2)
lgd=legend('$\mathcal{K}_0$','$\mathcal{K}_1$', 'Interpreter', 'latex');
lgd.Box='off';
lgd.FontSize = 15;
lgd.ItemTokenSize = [10,6];
box off
% 
fig = figure(8);
clf();
set(gcf,'Position',[309,513,600,270])

SS8=[K001(120:136)';...
    K010(120:136)'];

bar(1:17,SS8)
hold on
ylim([1e-3,1e6])
ylabel('$\mathcal{K}$', 'Interpreter', 'latex');
set(gca,'Ytick',[1e-3,1e0,1e3,1e6],'xtick',1:17,'xticklabel',52:68,'Yscale','log')
set(gca,'Yscale','log','FontName','Helvetica','FontSize',16,'FontWeight','bold','linewidth',1.2)
lgd=legend('$\mathcal{K}_0$','$\mathcal{K}_1$', 'Interpreter', 'latex');
lgd.Box='off';
lgd.FontSize = 15;
lgd.ItemTokenSize = [10,6];
box off