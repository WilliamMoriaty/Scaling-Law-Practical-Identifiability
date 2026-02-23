function [f, Jx, Jt, Hxx, Hxt, Htt] = get_hiv_derivatives(x, theta)
    % 变量映射
    T = x(1); I = x(2); V = x(3);
    
    % 参数映射
    lambda = theta(1); d = theta(2); k_param = theta(3); 
    delta = theta(4);  p = theta(5); c = theta(6);
    
    % === 1. f(x, theta) ===
    f = zeros(3,1);
    f(1) = lambda - d * T - k_param * V * T;
    f(2) = k_param * V * T - delta * I;
    f(3) = p * I - c * V;
    
    % === 2. Jx (df/dx) - 3x3 ===
    % Rows: f1, f2, f3
    % Cols: T, I, V
    Jx = zeros(3,3);
    
    % Row 1: lambda - d*T - k*V*T
    Jx(1,1) = -d - k_param * V; % df1/dT
    Jx(1,2) = 0;                % df1/dI
    Jx(1,3) = -k_param * T;     % df1/dV
    
    % Row 2: k*V*T - delta*I
    Jx(2,1) = k_param * V;      % df2/dT
    Jx(2,2) = -delta;           % df2/dI
    Jx(2,3) = k_param * T;      % df2/dV
    
    % Row 3: p*I - c*V
    Jx(3,1) = 0;
    Jx(3,2) = p;
    Jx(3,3) = -c;
    
    % === 3. Jt (df/dtheta) - 3x6 ===
    % Params: [lambda, d, k, delta, p, c]
    Jt = zeros(3, 6);
    
    % f1: lambda - d*T - k*V*T
    Jt(1,1) = 1;    % d/dlambda
    Jt(1,2) = -T;   % d/dd
    Jt(1,3) = -V*T; % d/dk
    % rest are 0
    
    % f2: k*V*T - delta*I
    Jt(2,3) = V*T;  % d/dk
    Jt(2,4) = -I;   % d/ddelta
    
    % f3: p*I - c*V
    Jt(3,5) = I;    % d/dp
    Jt(3,6) = -V;   % d/dc
    
    % === 4. 二阶导数初始化 ===
    % Hxx: 3个 3x3 矩阵 (对应 f1, f2, f3)
    Hxx = cell(3,1);
    Hxx{1} = zeros(3,3); Hxx{2} = zeros(3,3); Hxx{3} = zeros(3,3);
    
    % Hxt: 3个 3x6 矩阵 (State x Parameter)
    Hxt = cell(3,1);
    Hxt{1} = zeros(3,6); Hxt{2} = zeros(3,6); Hxt{3} = zeros(3,6);
    
    % Htt: 3个 6x6 矩阵 (Param x Param)
    Htt = cell(3,1);
    Htt{1} = zeros(6,6); Htt{2} = zeros(6,6); Htt{3} = zeros(6,6);
    
    % === 填写非零二阶导数 ===
    
    % --- Equation 1: f1 = lambda - d*T - k*V*T ---
    % Hxx (Variables: T, I, V)
    % d(-d-kV)/dV = -k
    Hxx{1}(1,3) = -k_param; % d2f1 / dT dV
    Hxx{1}(3,1) = -k_param; % d2f1 / dV dT
    
    % Hxt (Rows: T,I,V; Cols: lam, d, k, del, p, c)
    % df1/dT = -d - kV  -> deriv w.r.t d is -1, w.r.t k is -V
    Hxt{1}(1, 2) = -1;      % d2f1 / dT dd
    Hxt{1}(1, 3) = -V;      % d2f1 / dT dk
    % df1/dV = -kT      -> deriv w.r.t k is -T
    Hxt{1}(3, 3) = -T;      % d2f1 / dV dk
    
    % --- Equation 2: f2 = k*V*T - delta*I ---
    % Hxx
    Hxx{2}(1,3) = k_param;  % d2f2 / dT dV
    Hxx{2}(3,1) = k_param;  % d2f2 / dV dT
    
    % Hxt
    % df2/dT = kV       -> deriv w.r.t k is V
    Hxt{2}(1, 3) = V;       % d2f2 / dT dk
    % df2/dI = -delta   -> deriv w.r.t delta is -1
    Hxt{2}(2, 4) = -1;      % d2f2 / dI ddelta
    % df2/dV = kT       -> deriv w.r.t k is T
    Hxt{2}(3, 3) = T;       % d2f2 / dV dk
    
    % --- Equation 3: f3 = p*I - c*V ---
    % Hxx is all zero (linear state)
    
    % Hxt
    % df3/dI = p        -> deriv w.r.t p is 1
    Hxt{3}(2, 5) = 1;
    % df3/dV = -c       -> deriv w.r.t c is -1
    Hxt{3}(3, 6) = -1;
    
    % Htt (Param-Param) 都是 0，因为没有参数相乘的项 (e.g., no k*p)
end