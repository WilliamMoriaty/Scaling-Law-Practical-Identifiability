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
N2=50;
x = linspace(1, 4, N1);  % fit data
% x1 = linspace(0,4,N2); %uncertainty data

% 计算polynomial函数值 
y =zeros(N1,1);
for i = 1:N1
y(i) = poly_function(x(i), theta1,theta2,theta3,theta4);
end
% y1 =zeros(N2,1);
% for j = 1:N2
% y1(j) = poly_function(x1(j), theta1,theta2,theta3,theta4);
% end
% y1 = poly_function(x1, theta1,theta2,theta3,theta4);
yy=zeros(N1,4);
for i=1:N1
yy(i,:)=poly_para(x(i),theta1,theta2,theta3,theta4);
end
F = yy'*yy;
[U,Sigma,~]=svd(F);
% F = [17	-8	-12	-4;
% -8	4	6	2;
% -12	6	9	3;
% -4	2	3	1];

% [U,Sigma,~]=svd(F);
H = Hessian_matrix();
% H = [ 0,     0,      0,      0;
%       0,     2,     -1,     -1;
%       0,    -1,   -2,      0;
%       0,    -1,      0,   2];
delta = 1e-2;
% theta_1
P1 = [0 0 0 1;0 1 0 0;0 0 1 0;1 0 0 0];
F1 = P1'*F*P1;
H1 = P1'*H*P1;

F11 = F1(1:3,1:3);
[U1,Sigma1,~] = svd(F11);
if rank(Sigma1) < 3
nn = rank(Sigma1);
F12 = F1(1:3,4);
F21 = F1(4,1:3);
F22 = F1(4,4);
K01 = F22 - F21 * pinv(F11) * F12;
P = eye(4);
P(:,[1,4]) = P(:,[4,1]);
s1 = yy(:,1);
S = yy * P;
A = S(:,1:3);
K0_1 = norm((eye(4)-A*pinv(A))*s1,2)^2;

G1 = F21*U1(:,1:1);G2 = F21*U1(:,2:3);

H11 = H1(1:3,1:3);
H12 = H1(1:3,4);
H21 = H1(4,1:3);
H22 = H1(4,4);

K001 = F22 + delta*H22 - (F21+delta*H21)*pinv(F11+delta*H11) * (F21+delta*H21)';

K1 = H21*U1(:,1:1);K2 = H21*U1(:,2:3);
Sigma = Sigma1(1:nn,1:nn);
H11 = U1'*H11*U1;
h11 = H11(1:nn,1:nn);
h12 = H11(1:nn,nn+1:3);
h22 = H11(nn+1:3,nn+1:3);
K11 = H22 - (K1*pinv(Sigma)*G1'+G1*pinv(Sigma)*K1'+K2*pinv(h22)*K2'+G1*pinv(Sigma)*h12*pinv(h22)*h12'*pinv(Sigma)*G1')...
    +G1*pinv(Sigma)*h11*pinv(Sigma)*G1'+K2*pinv(h22)*h12'*pinv(Sigma)*G1'+G1*pinv(Sigma)*h12*pinv(h22)*K2';

term1 = K1*pinv(Sigma)*K1' - G1*pinv(Sigma)*h11*pinv(Sigma)*K1' - K1*pinv(Sigma)*h11*pinv(Sigma)*G1'...
    + G1*(pinv(Sigma)*h11*pinv(Sigma)*h11*pinv(Sigma))*G1';
term2 = (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12'*pinv(Sigma)*h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term3 = (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma)) * (h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term4 =  (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12') * (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma))';

K21 = -(term1 + term2 + term3 + term4);
else
F12 = F1(1:3,4);
F21 = F1(4,1:3);
F22 = F1(4,4);
K01 = F22 - F21 * pinv(F11) * F12;
G1 = F21*U1;

H11 = H1(1:3,1:3);
H12 = H1(1:3,4);
H21 = H1(4,1:3);
H22 = H1(4,4);
K001 = F22 + delta*H22 - (F21+delta*H21)*pinv(F11+delta*H11) * (F21+delta*H21)';
K1 = H21*U1;
M1 = F21*pinv(F11);
K11 = H22 - H21*pinv(F11)*F21' - F21*pinv(F11)*H21' + F21*pinv(F11)*H11*pinv(F11)*F21';
K21 = -H21*pinv(F11)*H21'+H21*pinv(F11)*H11*pinv(F11)*F21'+F21*pinv(F11)*H11*pinv(F11)*H21';

end


%% 
% 
% theta_2
P2 = [1 0 0 0;0 0 0 1;0 0 1 0;0 1 0 0];
F2 = P2'*F*P2;
H2 = P2'*H*P2;

F11 = F2(1:3,1:3);
[U2,Sigma2,~] = svd(F11);
if rank(Sigma2) < 3
nn = rank(Sigma2);
F12 = F2(1:3,4);
F21 = F2(4,1:3);
F22 = F2(4,4);
K02 = F22 - F21 * pinv(F11) * F12;

P = eye(4);
P(:,[2,4]) = P(:,[4,2]);
s2 = yy(:,2);
S = yy * P;
A = S(:,1:3);
K0_2 = norm((eye(4)-A*pinv(A))*s2,2)^2;


G1 = F21*U2(:,1:2);G2 = F21*U2(:,3);

H11 = H2(1:3,1:3);
H12 = H2(1:3,4);
H21 = H2(4,1:3);
H22 = H2(4,4);

K002 = F22 + delta*H22 - (F21+delta*H21)*pinv(F11+delta*H11) * (F21+delta*H21)';

K1 = H21*U2(:,1:2);K2 = H21*U2(:,3);
Sigma = Sigma2(1:nn,1:nn);
H11 = U2'*H11*U2;
h11 = H11(1:nn,1:nn);
h12 = H11(1:nn,nn+1);
h22 = H11(nn+1:3,nn+1:3);
K12 = H22 - (K1*pinv(Sigma)*G1'+G1*pinv(Sigma)*K1'+K2*pinv(h22)*K2'+G1*pinv(Sigma)*h12*pinv(h22)*h12'*pinv(Sigma)*G1')...
    +G1*pinv(Sigma)*h11*pinv(Sigma)*G1'+K2*pinv(h22)*h12'*pinv(Sigma)*G1'+G1*pinv(Sigma)*h12*pinv(h22)*K2';

term1 = K1*pinv(Sigma)*K1' - G1*pinv(Sigma)*h11*pinv(Sigma)*K1' - K1*pinv(Sigma)*h11*pinv(Sigma)*G1'...
    + G1*(pinv(Sigma)*h11*pinv(Sigma)*h11*pinv(Sigma))*G1';
term2 = (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12'*pinv(Sigma)*h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term3 = (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma)) * (h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term4 =  (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12') * (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma))';

K22 = -(term1 + term2 + term3 + term4);

else
F12 = F2(1:3,4);
F21 = F2(4,1:3);
F22 = F2(4,4);
K02 = F22 - F21 * pinv(F11) * F12;
G1 = F21*U2;

H11 = H2(1:3,1:3);
H12 = H2(1:3,4);
H21 = H2(4,1:3);
H22 = H2(4,4);
K002 = F22 + delta*H22 - (F21+delta*H21)*pinv(F11+delta*H11) * (F21+delta*H21)';
K1 = H21*U2;
M2 = F21*pinv(F11);
K12 = H22 - H21*pinv(F11)*F21' - F21*pinv(F11)*H21' + F21*pinv(F11)*H11*pinv(F11)*F21';
K22 = -H21*pinv(F11)*H21'+H21*pinv(F11)*H11*pinv(F11)*F21'+F21*pinv(F11)*H11*pinv(F11)*H21';

end


%%
% theta_3
P3 = [1 0 0 0;0 1 0 0;0 0 0 1;0 0 1 0];
F3 = P3'*F*P3;
H3 = P3'*H*P3;

F11 = F3(1:3,1:3);
[U3,Sigma3,~] = svd(F11);
if rank(Sigma3) < 3
nn = rank(Sigma3);
F12 = F3(1:3,4);
F21 = F3(4,1:3);
F22 = F3(4,4);
K03 = F22 - F21 * pinv(F11) * F12;

P = eye(4);
P(:,[3,4]) = P(:,[4,3]);
s3 = yy(:,3);
S = yy * P;
A = S(:,1:3);
K0_3 = norm((eye(4)-A*pinv(A))*s3,2)^2;


G1 = F21*U3(:,1:2);G2 = F21*U3(:,3);

H11 = H3(1:3,1:3);
H12 = H3(1:3,4);
H21 = H3(4,1:3);
H22 = H3(4,4);

K003 = F22 + delta*H22 - (F21+delta*H21)*pinv(F11+delta*H11) * (F21+delta*H21)';

K1 = H21*U3(:,1:2);K2 = H21*U3(:,3);
Sigma = Sigma3(1:nn,1:nn);
H11 = U3'*H11*U3;
h11 = H11(1:nn,1:nn);
h12 = H11(1:nn,nn+1);
h22 = H11(nn+1:3,nn+1:3);
K13 = H22 - (K1*pinv(Sigma)*G1'+G1*pinv(Sigma)*K1'+K2*pinv(h22)*K2'+G1*pinv(Sigma)*h12*pinv(h22)*h12'*pinv(Sigma)*G1')...
    +G1*pinv(Sigma)*h11*pinv(Sigma)*G1'+K2*pinv(h22)*h12'*pinv(Sigma)*G1'+G1*pinv(Sigma)*h12*pinv(h22)*K2';

term13 = K1*pinv(Sigma)*K1' - G1*pinv(Sigma)*h11*pinv(Sigma)*K1' - K1*pinv(Sigma)*h11*pinv(Sigma)*G1'...
    + G1*(pinv(Sigma)*h11*pinv(Sigma)*h11*pinv(Sigma))*G1';
term23 = (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12'*pinv(Sigma)*h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term33 = (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma)) * (h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term43 =  (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12') * (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma))';

K23 = -(term13 + term23 + term33 + term43);
else
F12 = F3(1:3,4);
F21 = F3(4,1:3);
F22 = F3(4,4);
K03 = F22 - F21 * pinv(F11) * F12;
G1 = F21*U3;

H11 = H3(1:3,1:3);
H12 = H3(1:3,4);
H21 = H3(4,1:3);
H22 = H3(4,4);
K003 = F22 + delta*H22 - (F21+delta*H21)*pinv(F11+delta*H11) * (F21+delta*H21)';
K1 = H21*U3;
M3 = F21*pinv(F11);
K13 = H22 - H21*pinv(F11)*F21' - F21*pinv(F11)*H21' + F21*pinv(F11)*H11*pinv(F11)*F21';
K23 = -H21*pinv(F11)*H21'+H21*pinv(F11)*H11*pinv(F11)*F21'+F21*pinv(F11)*H11*pinv(F11)*H21';

end

%%
% theta_4
P4 = [1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 1];
F4 = P4'*F*P4;
H4 = P4'*H*P4;
F11 = F4(1:3,1:3);
[U4,Sigma4,~] = svd(F11);

if rank(Sigma4) < 3
nn = rank(Sigma4);
F12 = F4(1:3,4);
F21 = F4(4,1:3);
F22 = F4(4,4);
K04 = F22 - F21 * pinv(F11) * F12;

P = eye(4);
s4 = yy(:,4);
S = yy * P;
A = S(:,1:3);
K0_4 = norm((eye(4)-A*pinv(A))*s4,2)^2;

G1 = F21*U4(:,1:2);G2 = F21*U4(:,3);

H11 = H4(1:3,1:3);
H11 = U4'*H11*U4;
H12 = H4(1:3,4);
H21 = H4(4,1:3);
H22 = H4(4,4);

K004 = F22 + delta*H22 - (F21+delta*H21)*pinv(F11+delta*H11) * (F21+delta*H21)';

K1 = H21*U4(:,1:2);K2 = H21*U4(:,3);
Sigma = Sigma4(1:nn,1:nn);
H11 = U4'*H11*U4;
h11 = H11(1:nn,1:nn);
h12 = H11(1:nn,nn+1);
h22 = H11(nn+1:3,nn+1:3);
K14 = H22 - (K1*pinv(Sigma)*G1'+G1*pinv(Sigma)*K1'+K2*pinv(h22)*K2'+G1*pinv(Sigma)*h12*pinv(h22)*h12'*pinv(Sigma)*G1')...
    +G1*pinv(Sigma)*h11*pinv(Sigma)*G1'+K2*pinv(h22)*h12'*pinv(Sigma)*G1'+G1*pinv(Sigma)*h12*pinv(h22)*K2';

term14 = K1*pinv(Sigma)*K1' - G1*pinv(Sigma)*h11*pinv(Sigma)*K1' - K1*pinv(Sigma)*h11*pinv(Sigma)*G1'...
    + G1*(pinv(Sigma)*h11*pinv(Sigma)*h11*pinv(Sigma))*G1';
term24 = (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12'*pinv(Sigma)*h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term34 = (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma)) * (h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term44 =  (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12') * (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma))';

K24 = -(term14 + term24 + term34 + term44);

else
F12 = F4(1:3,4);
F21 = F4(4,1:3);
F22 = F4(4,4);
K01 = F22 - F21 * pinv(F11) * F12;
G1 = F21*U4;

H11 = H4(1:3,1:3);
H12 = H4(1:3,4);
H21 = H4(4,1:3);
H22 = H4(4,4);
K004 = F22 + delta*H22 - (F21+delta*H21)*pinv(F11+delta*H11) * (F21+delta*H21)';
K1 = H21*U4;
M4 = F21*pinv(F11);
K14 = H22 - H21*pinv(F11)*F21' - F21*pinv(F11)*H21' + F21*pinv(F11)*H11*pinv(F11)*F21';
K24 = -H21*pinv(F11)*H21'+H21*pinv(F11)*H11*pinv(F11)*F21'+F21*pinv(F11)*H11*pinv(F11)*H21';

end


%%
% 
% figure(2)
% clf();
% set(gcf,'Position',[314,399,818,233])
% subplot(1,3,1)
% 
% SS1=[K01 K02 K03 K04];
% bar(1:4,SS1,'EdgeColor','none','FaceColor',[0,0,0])
% hold on
% ylim([0,2])
% ylabel('$$\mathcal{K}_0$', 'Interpreter', 'latex');
% set(gca,'xticklabel',{'\theta_1','\theta_2','\theta_3','\theta_4'},'Yscale','linear')
% set(gca,'FontName','Helvetica','FontSize',15,'FontWeight','bold','linewidth',1.2)
% box off
% 
% subplot(1,3,2)
% 
% SS2=[K11 K12 K13 K14];
% bar(1:4,SS2,'EdgeColor','none','FaceColor',[0,0,0])
% hold on
% ylim([1e-4,100])
% ylabel('$$\mathcal{K}_1$', 'Interpreter', 'latex');
% set(gca,'xticklabel',{'\theta_1','\theta_2','\theta_3','\theta_4'},'Yscale','log')
% set(gca,'FontName','Helvetica','FontSize',15,'FontWeight','bold','linewidth',1.2)
% box off
% 
% 
% subplot(1,3,3)
% 
% SS2=[K21 K22 K23 K24];
% bar(1:4,SS2,'EdgeColor','none','FaceColor',[0,0,0])
% hold on
% ylim([1e-4,1e0])
% ylabel('$$\mathcal{K}_2$', 'Interpreter', 'latex');
% set(gca,'xticklabel',{'\theta_1','\theta_2','\theta_3','\theta_4'},'Yscale','log')
% set(gca,'FontName','Helvetica','FontSize',15,'FontWeight','bold','linewidth',1.2)
% box off
%%
figure(3)
clf();
set(gcf,'Position',[529,357,361,235])


SS1=[K01 K02 K03 K04;...
    K11 K12 K13 K14;...
    K21 K22 K23 K24];

bar(1:4,SS1)
hold on
ylim([1e-4,100])
ylabel('$\mathcal{K}$', 'Interpreter', 'latex');
set(gca,'xticklabel',{'\theta_1','\theta_2','\theta_3','\theta_4'},'Yscale','log')
set(gca,'FontName','Helvetica','FontSize',16,'FontWeight','bold','linewidth',1.2)
lgd=legend('$\mathcal{K}_0$','$\mathcal{K}_1$','$\mathcal{K}_2$', 'Interpreter', 'latex');
lgd.Box='off';
lgd.FontSize = 15;
lgd.ItemTokenSize = [10,6];
box off

%%
thres = 1e-3;
% zero-order
[U0,Sigma_ff,~]=svd(F);
nnn0 = sum(diag(Sigma_ff)>thres);
H0 = H;
% first-order
U00 = U0(:,nnn0+1:4);
F1 = U00'*H*U00;
[U1,Sigma_ff1,~] = svd(F1);
nnn1 = sum(diag(Sigma_ff1)>thres);
U11 = [U0(:,1:2) U0(:,3:4)*U1];
% second-order
U10 = U1(:,2);
H1 = U00'*H0*U0(:,1:2)*pinv(Sigma_ff(1:2,1:2))*U0(:,1:2)'*H0*U00;
F2 = U10'*H1*U10;
[U2,Sigma_ff2,~] = svd(F2);
fig4=figure(4);
clf();
set(gcf,"Position",[328,106,652,478])

subplot(2,2,1)
bar(1:4,diag(Sigma_ff),'BarWidth',0.5)
hold on
plot([0.5,4.5],[thres thres],'k--','LineWidth',1.5)
ylim([1e-4,1e2])
xlim([0.5,4.5])
ylabel('Eigenvalue');
set(gca,'Ytick',[1e-4,1e-2,1e0,1e2],'xtick',1:4,'xticklabel',1:4,'Yscale','log')
set(gca,'Yscale','log','FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off
title('$\mathcal{O}(1)$','FontSize',18,'FontWeight','bold','Interpreter','latex')

subplot(2,2,2)
bar(1:2,diag(Sigma_ff1),'FaceColor',[0.85,0.33,0.10],'BarWidth',0.25)
hold on
plot([0.5,2.5],[thres thres],'k--','LineWidth',1.5)
xlim([0.5,2.5])
ylim([1e-4,1e2])
ylabel('Eigenvalue');
set(gca,'Ytick',[1e-4,1e-2,1e0,1e2],'xtick',1:2,'xticklabel',3:4,'Yscale','log')
set(gca,'Yscale','log','FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off

title('$\mathcal{O}(\varepsilon)$','FontSize',18,'FontWeight','bold','Interpreter','latex')


subplot(2,2,3)

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
c.FontSize = 14;  % 调整字体大小
c.Label.FontWeight = 'bold';  % 设置字体加粗
set(gca,'FontWeight','bold','FontSize',14)
set(gca,'XTick',1:4,'xticklabel',{'U_1','U_2','U_3','U_4'},...
    'YTick',1:4,'yticklabel',{'\theta_1','\theta_2','\theta_3','\theta_4'})

subplot(2,2,4)

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
c.FontSize = 14;  % 调整字体大小
c.Label.FontWeight = 'bold';  % 设置字体加粗
set(gca,'FontWeight','bold','FontSize',14)
set(gca,'XTick',1:4,'xticklabel',{'U_1','U_2','U_3','U_4'},...
    'YTick',1:4,'yticklabel',{'\theta_1','\theta_2','\theta_3','\theta_4'})

