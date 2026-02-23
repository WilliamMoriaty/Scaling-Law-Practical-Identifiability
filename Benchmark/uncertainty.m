% fig1=figure(1);
% clf();
% set(gcf,'Position',[780,134,275,678])
% 设定参数
clc;clear;
theta1 = 2; 
theta2 = 0;    
theta3 = 0;
theta4 = 0;
N1=4;
N2=101;
x = linspace(1, 4, N1);  % fit data
x1 = linspace(-1,6,N2); %uncertainty data

% 计算polynomial函数值 
y =zeros(N1,1);
for i = 1:N1
y(i) = poly_function(x(i), theta1,theta2,theta3,theta4);
end
y1 =zeros(N2,1);
for j = 1:N2
y1(j) = poly_function(x1(j), theta1,theta2,theta3,theta4);
end
yy=zeros(N1,4);
for i=1:N1
yy(i,:)=poly_para(x(i),theta1,theta2,theta3,theta4);
end
F = yy'*yy;
[U,Sigma,~]=svd(F);
% F = [15	-3	-6	-9;
% -3	1	2	3;
% -6	2	4	6;
% -9	3	6	9];

% [U,Sigma,~]=svd(F);
H = Hessian_matrix();
% H = diag([0,0.5,1e-5,1e-5]);

%%
thres = 1e-6;
% zero-order
[U0,Sigma_ff,~]=svd(F);
nnn0 = sum(diag(Sigma_ff)>thres);
H0 = H;
% first-order
U00 = U0(:,nnn0+1:4);
F1 = U00'*H*U00;
[U1,Sigma_ff1,~] = svd(F1);
nnn1 = sum(diag(Sigma_ff1)>thres);
U11 = [U0(:,1:nnn0) U0(:,nnn0+1:4)*U1];
% second-order
U10 = U1(:,nnn1+1:4-nnn0);
H1 = U00'*H0*U0(:,1:nnn0)*pinv(Sigma_ff(1:nnn0,1:nnn0))*U0(:,1:nnn0)'*H0*U00;
F2 = U10'*H1*U10;
[U2,Sigma_ff2,~] = svd(F2);
H2 =  U10'*H1*U1(:,1)*pinv(Sigma_ff1(1,1))*U1(:,1)'*H1*U10;
fig4=figure(4);
clf();
set(gcf,"Position",[322,296,680,217])
subplot(1,2,1)

alphaData = ones(4,4);  % 初始化为全不透明
alphaData(:, 3:4) = 0.2;  % 设置右半部分透明度为 0.2

imagesc(abs(U0),'AlphaData',alphaData);

% 设置 colormap
cmap = othercolor('BuDRd_12');
colormap(cmap);  % 可以选择其他 colormap 例如 'jet', 'hot', 'cool' 等
clim([0,1.1])
% 添加 colorbar 并设置标签
c = colorbar;
c.Label.String = '|\partial U_i^T\theta/\partial \theta_j|';  % 设置 colorbar 的标签
c.FontSize = 12;  % 调整字体大小
c.Label.FontWeight = 'bold';  % 设置字体加粗
set(gca,'FontWeight','bold','FontSize',12)
set(gca,'XTick',1:4,'xticklabel',{'U_1','U_2','U_3','U_4'},...
    'YTick',1:4,'yticklabel',{'\theta_1','\theta_2','\theta_3','\theta_4'})
title('$\mathcal{O}(1)$','FontSize',14,'FontWeight','bold','Interpreter','latex')

subplot(1,2,2)

alphaData1 = ones(4,4);  % 初始化为全不透明
alphaData1(:, 4:4) = 0.2;  % 设置右半部分透明度为 0.2

imagesc(1:4,1:4,abs(U11),'AlphaData',alphaData1);

% 设置 colormap
cmap = othercolor('BuDRd_12');
colormap(cmap);  % 可以选择其他 colormap 例如 'jet', 'hot', 'cool' 等
clim([0,1.1])
% 添加 colorbar 并设置标签
c = colorbar;
c.Label.String = '|\partial U_i^T\theta/\partial \theta_j|';  % 设置 colorbar 的标签
c.FontSize = 12;  % 调整字体大小
c.Label.FontWeight = 'bold';  % 设置字体加粗
set(gca,'FontWeight','bold','FontSize',12)
set(gca,'XTick',1:4,'xticklabel',{'U_1','U_2','U_3','U_4'},...
    'YTick',1:4,'yticklabel',{'\theta_1','\theta_2','\theta_3','\theta_4'})
title('$\mathcal{O}(\varepsilon)$','FontSize',14,'FontWeight','bold','Interpreter','latex')

%%
Var = zeros(N2,1);
thresh = 0.01;
r = min(find(diag(Sigma)<thresh));
for i=1:N2
yy1 = poly_para(x1(i),theta1,theta2,theta3,theta4);

Var(i) = yy1*U0(:,r:end)*U0(:,r:end)'*yy1';
end
% 
 Var1 = zeros(N2,1);
r1 = 4;
for i=1:N2
yy1 = poly_para(x1(i),theta1,theta2,theta3,theta4);

Var1(i) = yy1*U11(:,r1:end)*U11(:,r1:end)'*yy1';
end
% xlabel('t');
% ylabel('h(t,\theta)')
% %ylabel({'$\varphi(t,\theta)$'},'Interpreter','latex');
% lgd = legend("Data","Polynomial","95% CI");
% lgd.FontWeight = 'bold';
% lgd.Location = 'best';
% lgd.Box='off';
% lgd.ItemTokenSize = [10,6];
% set(gca,'FontName','Helvetica','FontSize',15,'FontWeight','bold','linewidth',1.2)
% box off



fig1=figure(1);
clf();
set(gcf,"Position",[430,201,543,398])
% subplot(2,2,1)
plot(x,y,'Marker','o','Color','r','LineStyle','none','MarkerSize',10,'LineWidth',2.0)
hold on

sigma_th=1e-1;
Y_fit=y1(1:N2);

CI=1.96*sqrt(sigma_th*Var);
% 绘制置信区域（拟合曲线上下的置信区间）
fill([x1'; flipud(x1')], [Y_fit + CI; flipud(Y_fit - CI)], 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
hold on
sigma_th=1e-1;
CI1=1.96*sqrt(sigma_th*Var1);
% 绘制置信区域（拟合曲线上下的置信区间）
fill([x1'; flipud(x1')], [Y_fit + CI1; flipud(Y_fit - CI1)],'g', 'FaceAlpha', 0.6, 'EdgeColor', 'none');
hold on
plot(x1,y1,'LineStyle','-','Color','k','LineWidth',2.0)
hold on
xlim([0.5,4.5])
xlabel('t')
ylabel('h')
set(gca,'FontWeight','bold','FontSize',20,'LineWidth',2.0)
lgd = legend('$\mathbf{data}$', ...
             '$\mathcal{O}(1)$', ...
             '$\mathcal{O}(\varepsilon)$', ...
             '$\mathbf{model}$', ...
             'Interpreter', 'latex');
lgd.FontSize = 18;
lgd.Location = 'northeast';
lgd.ItemTokenSize = [10,6];
lgd.Box = 'off';
box off

% subplot(2,2,2)
% plot(x1,CI-CI1,'LineStyle','-','Color','k','LineWidth',1.5)
% xlabel('t')
% ylabel('Var_0-Var_1')
% set(gca,'YScale','log')
% % ylim([1e-6,0.06])
% xlim([0.5,4.5])
% box off
% 
% subplot(2,2,4)
% plot(x,y,'Marker','o','Color','r','LineStyle','none','MarkerSize',10)
% hold on
% plot(x1,y1,'LineStyle','-','Color','k','LineWidth',1.5)
% hold on
% 
% Y_fit=y1(1:N2);
% CI1=1.96*sqrt(sigma_th*Var1);
% % 绘制置信区域（拟合曲线上下的置信区间）
% fill([x1'; flipud(x1')], [Y_fit + CI1; flipud(Y_fit - CI1)],'g', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% hold on
% CI=1.96*sqrt(sigma_th*Var);
% % % 绘制置信区域（拟合曲线上下的置信区间）
% % fill([x1'; flipud(x1')], [Y_fit + CI; flipud(Y_fit - CI)], 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% % hold on
% xlim([0.5,4.5])
% xlabel('t')
% ylabel('y')
% set(gca,'FontWeight','bold','FontSize',12)
% title('$\mathcal{O}(\varepsilon)$','FontSize',14,'FontWeight','bold','Interpreter','latex')
% box off
% 
% subplot(2,2,3)
% plot(x,y,'Marker','o','Color','r','LineStyle','none','MarkerSize',10)
% hold on
% plot(x1,y1,'LineStyle','-','Color','k','LineWidth',1.5)
% hold on
% xlim([0.5,4.5])
% Y_fit=y1(1:N2);
% % CI1=1.96*sqrt(sigma_th*Var1);
% % % 绘制置信区域（拟合曲线上下的置信区间）
% % fill([x1'; flipud(x1')], [Y_fit + CI1; flipud(Y_fit - CI1)],'g', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% % hold on
% CI=1.96*sqrt(sigma_th*Var);
% % 绘制置信区域（拟合曲线上下的置信区间）
% fill([x1'; flipud(x1')], [Y_fit + CI; flipud(Y_fit - CI)], 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% hold on
% 
% xlabel('t')
% ylabel('y')
% set(gca,'FontWeight','bold','FontSize',12)
% title('$\mathcal{O}(1)$','FontSize',14,'FontWeight','bold','Interpreter','latex')
% box off
% 
