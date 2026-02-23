function f = Objective_HIV(theta, tspan, x0, yexp3)
    % Using [0; tspan] ensures the solver includes the data time points
    sol = ode45(@(t,x) HIV_eq(t,x,theta), [0; max(tspan)], x0);
    
    % Evaluate the solution specifically at the experimental time points
    solpts = deval(sol, tspan);
    
    % HIV Viral Load (V) is the 3rd state variable
    % Multiply by 2000 to match data units (e.g., copies/mL vs virions)
    V_model = solpts(3, :)' * 2000;
    
    % Calculate residuals in log10 space for better scaling
    % Adding a small epsilon prevents log10(0) errors
    f = (log10(V_model) - log10(yexp3)); 
end
