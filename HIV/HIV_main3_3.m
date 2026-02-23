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

%%
fig = figure(1);
clf();
set(gcf,"Position",[478,414,408,272])
plot(t,log10(2*X1(:,3)*1e3),'LineWidth',1.5,'Color','k');
hold on
plot(tspan,log10(xexp3),'Marker','o','MarkerSize',10, ...
    'MarkerFaceColor','r','MarkerEdgeColor','r','LineStyle','none')
xlim([0,120])
ylim([2,7])
set(gca,'Yscale','linear')
xlabel('Day post infection (days)')
ylabel('log_{10} HIV (copies/ml)')
set(gca,'FontName','Helvetica','FontSize',15,'FontWeight','bold','linewidth',1.2)
lgd = legend("simu.","data");
lgd.ItemTokenSize = [15,6];
% lgd.Position = [0.746 0.301 0.139 0.326];
lgd.Location = 'northeast';
lgd.Box = 'off';
lgd.FontWeight = 'bold';
box off


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
NNN = size(data,1)+1;
% theta_1
P = eye(6);
P(:,[1,6]) = P(:,[6,1]);
P1 = P;
F1 = P1'*F*P1;
H1 = P1'*H*P1;

s1 = y_hat(:,1);
S = y_hat * P1;
A = S(:,1:5);
K0_1 = norm((eye(NNN)-A*pinv(A))*s1,2)^2;

F11 = F1(1:5,1:5);
[U1,Sigma1,~] = svd(F11);

nn1 = rank(Sigma1);
F12 = F1(1:5,6);
F21 = F1(6,1:5);
F22 = F1(6,6);
K01 = F22 - F21 * pinv(F11,1e2) * F12;

G1 = F21*U1(:,1:nn1);G2 = F21*U1(:,nn1+1:5);
% 
H11 = H1(1:5,1:5);
H12 = H1(1:5,6);
H21 = H1(6,1:5);
H22 = H1(6,6);
% 
 
K1 = H21*U1(:,1:nn1);K2 = H21*U1(:,nn1+1:5);
Sigma = Sigma1(1:nn1,1:nn1);
H11 = U1'*H11*U1;
h11 = H11(1:nn1,1:nn1);
h12 = H11(1:nn1,nn1+1:5);
h22 = H11(nn1+1:5,nn1+1:5);
K11 = H22 - (K1*pinv(Sigma)*G1'+G1*pinv(Sigma)*K1'+K2*pinv(h22)*K2'...
    +G1*pinv(Sigma)*h12*pinv(h22)*h12'*pinv(Sigma)*G1')...
    +G1*pinv(Sigma)*h11*pinv(Sigma)*G1'+K2*pinv(h22)*h12'*pinv(Sigma)*G1'...
    +G1*pinv(Sigma)*h12*pinv(h22)*K2';

term1 = K1*pinv(Sigma)*K1' - G1*pinv(Sigma)*h11*pinv(Sigma)*K1' - K1*pinv(Sigma)*h11*pinv(Sigma)*G1'...
    + G1*(pinv(Sigma)*h11*pinv(Sigma)*h11*pinv(Sigma))*G1';
term2 = (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12'*pinv(Sigma)*h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term3 = (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma)) * (h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term4 =  (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12') * (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma))';

K21 = -(term1 + term2 + term3 + term4);

% theta_2
P = eye(6);
P(:,[2,6]) = P(:,[6,2]);
P2 = P;
F2 = P2'*F*P2;
H2 = P2'*H*P2;

s2 = y_hat(:,2);
S = y_hat * P2;
A = S(:,1:5);
K0_2 = norm((eye(NNN)-A*pinv(A))*s2,2)^2;

F11 = F2(1:5,1:5);
[U2,Sigma2,~] = svd(F11);

nn2 = rank(Sigma2);
F12 = F2(1:5,6);
F21 = F2(6,1:5);
F22 = F2(6,6);
K02 = F22 - F21 * pinv(F11,1e2) * F12;

G1 = F21*U2(:,1:nn2);G2 = F21*U2(:,nn2+1:5);
% 
H11 = H2(1:5,1:5);
H12 = H2(1:5,6);
H21 = H2(6,1:5);
H22 = H2(6,6);
% 
 
K1 = H21*U2(:,1:nn2);K2 = H21*U2(:,nn2+1:5);
Sigma = Sigma2(1:nn2,1:nn2);
H11 = U2'*H11*U2;
h11 = H11(1:nn2,1:nn2);
h12 = H11(1:nn2,nn2+1:5);
h22 = H11(nn2+1:5,nn2+1:5);
K12 = H22 - (K1*pinv(Sigma)*G1'+G1*pinv(Sigma)*K1'+K2*pinv(h22)*K2'...
    +G1*pinv(Sigma)*h12*pinv(h22)*h12'*pinv(Sigma)*G1')...
    +G1*pinv(Sigma)*h11*pinv(Sigma)*G1'...
    +K2*pinv(h22)*h12'*pinv(Sigma)*G1'+G1*pinv(Sigma)*h12*pinv(h22)*K2';

term1 = K1*pinv(Sigma)*K1' - G1*pinv(Sigma)*h11*pinv(Sigma)*K1' - K1*pinv(Sigma)*h11*pinv(Sigma)*G1'...
    + G1*(pinv(Sigma)*h11*pinv(Sigma)*h11*pinv(Sigma))*G1';
term2 = (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12'*pinv(Sigma)*h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term3 = (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma)) * (h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term4 =  (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12') * (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma))';

K22 = -(term1 + term2 + term3 + term4);

% theta_3
P = eye(6);
P(:,[3,6]) = P(:,[6,3]);

P3 = P;
F3 = P3'*F*P3;
H3 = P3'*H*P3;

s3 = y_hat(:,3);
S = y_hat * P3;
A = S(:,1:5);
K0_3 = norm((eye(NNN)-A*pinv(A))*s3,2)^2;

F11 = F3(1:5,1:5);
[U3,Sigma3,~] = svd(F11);

nn3 = rank(Sigma3);
F12 = F3(1:5,6);
F21 = F3(6,1:5);
F22 = F3(6,6);
K03 = F22 - F21 * pinv(F11,1e1) * F12;

G1 = F21*U3(:,1:nn3);G2 = F21*U3(:,nn3+1:5);
% 
H11 = H3(1:5,1:5);
H12 = H3(1:5,6);
H21 = H3(6,1:5);
H22 = H3(6,6);
% 
 
K1 = H21*U3(:,1:nn3);K2 = H21*U3(:,nn3+1:5);
Sigma = Sigma3(1:nn3,1:nn3);
H11 = U3'*H11*U3;
h11 = H11(1:nn3,1:nn3);
h12 = H11(1:nn3,nn3+1:5);
h22 = H11(nn3+1:5,nn3+1:5);
K13 = H22 - (K1*pinv(Sigma)*G1'+G1*pinv(Sigma)*K1'+K2*pinv(h22)*K2'...
    +G1*pinv(Sigma)*h12*pinv(h22)*h12'*pinv(Sigma)*G1')...
    +G1*pinv(Sigma)*h11*pinv(Sigma)*G1'...
    +K2*pinv(h22)*h12'*pinv(Sigma)*G1'+G1*pinv(Sigma)*h12*pinv(h22)*K2';

term1 = K1*pinv(Sigma)*K1' - G1*pinv(Sigma)*h11*pinv(Sigma)*K1' - K1*pinv(Sigma)*h11*pinv(Sigma)*G1'...
    + G1*(pinv(Sigma)*h11*pinv(Sigma)*h11*pinv(Sigma))*G1';
term2 = (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12'*pinv(Sigma)*h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term3 = (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma)) * (h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term4 =  (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12') * (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma))';

K23 = -(term1 + term2 + term3 + term4);
% theta_4
P = eye(6);
P(:,[4,6]) = P(:,[6,4]);
P4 = P;
F4 = P4'*F*P4;
H4 = P4'*H*P4;

s4 = y_hat(:,4);
S = y_hat * P4;
A = S(:,1:5);
K0_4 = norm((eye(NNN)-A*pinv(A))*s4,2)^2;

F11 = F4(1:5,1:5);
[U4,Sigma4,~] = svd(F11);

nn4 = rank(Sigma4);
F12 = F4(1:5,6);
F21 = F4(6,1:5);
F22 = F4(6,6);
K04 = F22 - F21 * pinv(F11,1e1) * F12;

G1 = F21*U4(:,1:nn4);G2 = F21*U4(:,nn4+1:5);
% 
H11 = H4(1:5,1:5);
H12 = H4(1:5,6);
H21 = H4(6,1:5);
H22 = H4(6,6);
% 

K1 = H21*U4(:,1:nn4);K2 = H21*U4(:,nn4+1:5);
Sigma = Sigma4(1:nn4,1:nn4);
H11 = U4'*H11*U4;
h11 = H11(1:nn4,1:nn4);
h12 = H11(1:nn4,nn4+1:5);
h22 = H11(nn4+1:5,nn4+1:5);

K14 = H22 - (K1*pinv(Sigma)*G1'+G1*pinv(Sigma)*K1'...
    +K2*pinv(h22)*K2'+G1*pinv(Sigma)*h12*pinv(h22)*h12'*pinv(Sigma)*G1')...
    +G1*pinv(Sigma)*h11*pinv(Sigma)*G1'+...
    K2*pinv(h22)*h12'*pinv(Sigma)*G1'+G1*pinv(Sigma)*h12*pinv(h22)*K2';

term1 = K1*pinv(Sigma)*K1' - G1*pinv(Sigma)*h11*pinv(Sigma)*K1' - K1*pinv(Sigma)*h11*pinv(Sigma)*G1'...
    + G1*(pinv(Sigma)*h11*pinv(Sigma)*h11*pinv(Sigma))*G1';
term2 = (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12'*pinv(Sigma)*h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term3 = (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma)) * (h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term4 =  (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12') * (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma))';

K24 = -(term1 + term2 + term3 + term4);
% theta_5
P = eye(6);
P(:,[5,6]) = P(:,[6,5]);
P5 = P;
F5 = P5'*F*P5;
H5 = P5'*H*P5;

s5 = y_hat(:,5);
S = y_hat * P5;
A = S(:,1:5);
K0_5 = norm((eye(NNN)-A*pinv(A))*s5,2)^2;

F11 = F5(1:5,1:5);
[U5,Sigma5,~] = svd(F11);

nn5 = rank(Sigma5);
F12 = F5(1:5,6);
F21 = F5(6,1:5);
F22 = F5(6,6);
K05 = F22 - F21 * pinv(F11,1e1) * F12;

G1 = F21*U5(:,1:nn5);G2 = F21*U5(:,nn5+1:5);

H11 = H5(1:5,1:5);
H12 = H5(1:5,6);
H21 = H5(6,1:5);
H22 = H5(6,6);


K1 = H21*U5(:,1:nn5);K2 = H21*U5(:,nn5+1:5);
Sigma = Sigma5(1:nn5,1:nn5);
H11 = U5'*H11*U5;
h11 = H11(1:nn5,1:nn5);
h12 = H11(1:nn5,nn5+1:5);
h22 = H11(nn5+1:5,nn5+1:5);

K15 = H22 - (K1*pinv(Sigma)*G1'+G1*pinv(Sigma)*K1'...
    +K2*pinv(h22)*K2'+G1*pinv(Sigma)*h12*pinv(h22)*h12'*pinv(Sigma)*G1')...
    +G1*pinv(Sigma)*h11*pinv(Sigma)*G1'+K2*pinv(h22)*h12'*pinv(Sigma)*G1'...
    +G1*pinv(Sigma)*h12*pinv(h22)*K2';

term1 = K1*pinv(Sigma)*K1' - G1*pinv(Sigma)*h11*pinv(Sigma)*K1' - K1*pinv(Sigma)*h11*pinv(Sigma)*G1'...
    + G1*(pinv(Sigma)*h11*pinv(Sigma)*h11*pinv(Sigma))*G1';
term2 = (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12'*pinv(Sigma)*h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term3 = (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma)) * (h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term4 =  (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12') * (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma))';

K25 = -(term1 + term2 + term3 + term4);

% theta_6
P6 = eye(6);

F6 = P6'*F*P6;
H6 = P6'*H*P6;

s6 = y_hat(:,6);
S = y_hat * P6;
A = S(:,1:5);
K0_6 = norm((eye(NNN)-A*pinv(A))*s6,2)^2;

F11 = F6(1:5,1:5);
[U6,Sigma6,~] = svd(F11);

nn6 = rank(Sigma6);
F12 = F6(1:5,6);
F21 = F6(6,1:5);
F22 = F6(6,6);
K06 = F22 - F21 * pinv(F11,1e1) * F12;

G1 = F21*U6(:,1:nn6);G2 = F21*U6(:,nn6+1:5);

H11 = H6(1:5,1:5);
H12 = H6(1:5,6);
H21 = H6(6,1:5);
H22 = H6(6,6);


K1 = H21*U6(:,1:nn6);K2 = H21*U6(:,nn6+1:5);
Sigma = Sigma6(1:nn6,1:nn6);
H11 = U6'*H11*U6;
h11 = H11(1:nn6,1:nn6);
h12 = H11(1:nn6,nn6+1:5);
h22 = H11(nn6+1:5,nn6+1:5);

K16 = H22 - (K1*pinv(Sigma)*G1'+G1*pinv(Sigma)*K1'...
    +K2*pinv(h22)*K2'+G1*pinv(Sigma)*h12*pinv(h22)*h12'*pinv(Sigma)*G1')...
    +G1*pinv(Sigma)*h11*pinv(Sigma)*G1'+K2*pinv(h22)*h12'*pinv(Sigma)*G1'...
    +G1*pinv(Sigma)*h12*pinv(h22)*K2';

term1 = K1*pinv(Sigma)*K1' - G1*pinv(Sigma)*h11*pinv(Sigma)*K1' - K1*pinv(Sigma)*h11*pinv(Sigma)*G1'...
    + G1*(pinv(Sigma)*h11*pinv(Sigma)*h11*pinv(Sigma))*G1';
term2 = (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12'*pinv(Sigma)*h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term3 = (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma)) * (h12*pinv(h22)) * (G1*pinv(Sigma)*h12-K2)';
term4 =  (G1*pinv(Sigma)*h12-K2) * (pinv(h22)*h12') * (K1*pinv(Sigma)-G1*pinv(Sigma)*h11*pinv(Sigma))';

K26 = -(term1 + term2 + term3 + term4);
%%
fig = figure(2);
clf();
set(gcf,'Position',[529,263,416,329])


SS1=[K01 K02 K03 K04 K05 K06;...
    K11 K12 K13 K14 K15 K16];

bar(1:6,SS1)
hold on
xlim([0.5,6.5])
ylim([1e-3,1e9])
ylabel('$\mathcal{K}$', 'Interpreter', 'latex');
set(gca,'Ytick',[1e-3,1e0,1e3,1e6,1e9],'xticklabel',{'\lambda','d','k','\delta','\pi','c'},'Yscale','log')
set(gca,'FontName','Helvetica','FontSize',16,'FontWeight','bold','linewidth',1.2)
lgd=legend('$\mathcal{K}_0$','$\mathcal{K}_1$', 'Interpreter', 'latex');
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
U00 = U0(:,nnn0+1:6);
F1 = U00'*H*U00;
[U1,Sigma_ff1,~] = svd(F1);
nnn1 = sum(diag(Sigma_ff1)>thres);
U11 = [U0(:,1:nnn0) U0(:,nnn0+1:6)*U1];
% % second-order
% U10 = U1(:,2);
% H1 = U00'*H0*U0(:,1:2)*pinv(Sigma_ff(1:2,1:2))*U0(:,1:2)'*H0*U00;
% F2 = U10'*H1*U10;
% [U2,Sigma_ff2,~] = svd(F2);
fig3=figure(3);
clf();
set(gcf,"Position",[385,205,638,470])

subplot(2,2,1)
bar(1:6,diag(Sigma_ff),'BarWidth',0.5)
hold on
plot([0.5,6.5],[thres thres],'k--','LineWidth',1.5)
ylim([1e-4,1e8])
xlim([0.5,6.5])
ylabel('Eigenvalue');
xlabel('U');
set(gca,'Ytick',[1e-4,1e0,1e4,1e8],'xtick',1:6,'xticklabel',1:6,'Yscale','log')
set(gca,'Yscale','log','FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off
title('$\mathcal{O}(1)$','FontSize',18,'FontWeight','bold','Interpreter','latex')

subplot(2,2,2)
bar(1:3,diag(Sigma_ff1),'FaceColor',[0.85,0.33,0.10],'BarWidth',0.25)
hold on
plot([0.5,3.5],[thres thres],'k--','LineWidth',1.5)
xlim([0.5,3.5])
ylim([1e-4,1e8])
ylabel('Eigenvalue');
xlabel('U');
set(gca,'Ytick',[1e-4,1e0,1e4,1e8],'xtick',1:3,'xticklabel',4:6,'Yscale','log')
set(gca,'Yscale','log','FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off

title('$\mathcal{O}(\varepsilon)$','FontSize',18,'FontWeight','bold','Interpreter','latex')

subplot(2,2,3)

alphaData = ones(6,6);  % 初始化为全不透明
alphaData(:, nnn0+1:6) = 0.2;  % 设置右半部分透明度为 0.2

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
set(gca,'FontWeight','bold','FontSize',12)
set(gca,'XTick',1:6,'xticklabel',{'U_1','U_2','U_3','U_4','U_5','U_6'},...
    'YTick',1:6,'yticklabel',{'\lambda','d','k','\delta','\pi','c'})

subplot(2,2,4)

alphaData1 = ones(6,6);  % 初始化为全不透明
alphaData1(:, nnn0+nnn1+1:6) = 0.2;  % 设置右半部分透明度为 0.2

imagesc(1:6,1:6,abs(U11),'AlphaData',alphaData1);

% 设置 colormap
cmap = othercolor('BuDRd_12');
colormap(cmap);  % 可以选择其他 colormap 例如 'jet', 'hot', 'cool' 等
clim([0,1.1])
% 添加 colorbar 并设置标签
c = colorbar;
c.Label.String = '|\partial U_i^T\theta/\partial \theta_j|';  % 设置 colorbar 的标签
c.FontSize = 14;  % 调整字体大小
c.Label.FontWeight = 'bold';  % 设置字体加粗
set(gca,'FontWeight','bold','FontSize',12)
set(gca,'XTick',1:6,'xticklabel',{'U_1','U_2','U_3','U_4','U_5','U_6'},...
    'YTick',1:6,'yticklabel',{'\lambda','d','k','\delta','\pi','c'})


