# Overview of the Computational Approach to Parameter Identifiability Scaling Laws

We consider a general model $\boldsymbol{\varphi(t, \theta)}$, where $\boldsymbol{t}$ denotes the independent input variable (e.g., time) and $\boldsymbol{\theta} \in \mathbb{R}^k$ is the parameter vector. The model architecture is versatile, encompassing explicit functional forms (e.g., neural networks) or solutions to complex differential equations, with observable variables $\boldsymbol{h(\varphi(t, \theta))}$ mapped to experimental measurements $\{\boldsymbol{t}_i, \hat{\boldsymbol{h}}_i\}_{i=1}^N$ collected at discrete time points.

Using the least-squares objective
$
l(\boldsymbol{h}(t, \boldsymbol{\theta}), \hat{\boldsymbol{h}}) = \sum_{i=1}^N \| \boldsymbol{h}(t_i, \boldsymbol{\theta}) - \hat{\boldsymbol{h}}_i \|_2^2,
$
we obtain the optimal parameter set $\boldsymbol{\theta^*}$ and compute the Fisher Information Matrix (FIM, $F$) and the perturbed Hessian ($H$) \cite{wang2025systematic}. Our framework integrates eigenvalue decomposition (EVD) with the Schur complement (SC) to classify parameter identifiability across hierarchical scales. The algorithm is illustrated in {\bf Algorithm~\ref{am:1}}, and a detailed mathematical formulation is provided in {\bf Materials and Methods}.

We introduce a novel metric, $\mathcal{K}_i$, to quantify the $i$-th order of parameter identifiability. The zero-order component, $\mathcal{K}_0$, recovers traditional parameter identifiability,
$
\mathcal{K}_0 = \|(I - A A^{\dagger}) s_i \|_2^2,
$
\cite{wang2025systematic}, while the first-order metric, $\mathcal{K}_1$, captures the emergence of flat likelihood profiles when $\mathcal{K}_0 = 0$. A detailed analytical formulation of $\mathcal{K}_i$ is provided in {\bf Algorithm~\ref{am:1}}.

Furthermore, our higher-order uncertainty quantification (UQ) framework evaluates predictive uncertainty arising from non-identifiable subspaces. Specifically, it isolates contributions from zero-order non-identifiable parameters ($\boldsymbol{U_{k-r_0}^\top \theta}$, red region) and first-order non-identifiable parameters ($\boldsymbol{U_{k-r_0-r_1}^\top \theta}$, blue region) in Fig. \ref{fig:1}, thereby also characterizing the uncertainty order of the loss function.
<img width="4818" height="2992" alt="Fig1" src="https://github.com/user-attachments/assets/7ae94ea7-8bb9-4082-b773-549222774258" />

