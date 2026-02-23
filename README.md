# Overview of the Computational Approach to Parameter Identifiability Scaling Laws

We consider a general model $\mathbf{\varphi(t, \theta)}$, where $t$ denotes the independent input variable (e.g., time) and $\mathbf{\theta} \in \mathbb{R}^k$ is the parameter vector. The model architecture is versatile, encompassing explicit functional forms (e.g., neural networks) or solutions to complex differential equations, with observable variables $\mathbf{h(\varphi(t,\theta))}$ mapped to experimental measurements collected at discrete time points.

## 1. Objective Function & Optimization
Using the least-squares objective:

$$
l(\mathbf{h}(t, \mathbf{\theta}), \hat{\mathbf{h}}) = \sum_{i=1}^N \| \mathbf{h}(t_i, \mathbf{\theta}) - \hat{\mathbf{h}}_i \|_2^2
$$

We obtain the optimal parameter set $\mathbf{\theta^*}$ and compute the **Fisher Information Matrix (FIM, $F$)** and the **perturbed Hessian ($H$)**. Our framework integrates **eigenvalue decomposition (EVD)** with the **Schur complement (SC)** to classify parameter identifiability across hierarchical scales. 

---

## 2. Hierarchical Identifiability Metrics ($\mathcal{K}_i$)
We introduce a novel metric, $\mathcal{K}_i$, to quantify the $i$-th order of parameter identifiability:

* **Zero-order component ($\mathcal{K}_0$):** Recovers traditional parameter identifiability:
    $$\mathcal{K}_0 = \|(I - A A^{\dagger}) s_i \|_2^2$$
* **First-order metric ($\mathcal{K}_1$):** Captures the emergence of flat likelihood profiles when $\mathcal{K}_0 = 0$. 


---

## 3. Higher-Order Uncertainty Quantification (UQ)
Our higher-order UQ framework evaluates predictive uncertainty arising from non-identifiable subspaces. As shown in **Figure 1** below, it isolates contributions from:

* **Zero-order non-identifiable parameters:** ($\mathbf{U_{k-r_0}^\top \theta}$, <font color="red">red region</font>)
* **First-order non-identifiable parameters:** ($\mathbf{U_{k-r_0-r_1}^\top \theta}$, <font color="blue">blue region</font>)

This characterizes the **uncertainty order** of the loss function.

### Figure 1: Illustration of the UQ Framework
<img width="4818" height="2992" alt="Fig1" src="https://github.com/user-attachments/assets/5830de8d-c60d-4093-af62-582c98b59c77" />

---
