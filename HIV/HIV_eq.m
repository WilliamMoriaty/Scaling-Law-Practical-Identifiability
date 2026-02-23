function dxdt = HIV_eq(t, x, theta)
    % x(1): Target Cells (T)
    % x(2): Infected Cells (I)
    % x(3): Free Virions (V)
    
    lambda = theta(1); % Production of T cells
    d      = theta(2); % Death rate of T cells
    k      = theta(3); % Infection rate
    delta  = theta(4); % Death rate of I cells
    p      = theta(5); % Production rate of V
    c      = theta(6); % Clearance rate of V
    
    dxdt = [
        lambda - d*x(1) - k*x(3)*x(1); % dT/dt
        k*x(3)*x(1) - delta*x(2);      % dI/dt
        p*x(2) - c*x(3)                % dV/dt
    ];
end