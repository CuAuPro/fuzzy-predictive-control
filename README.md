# fuzzy-predictive-control
The Fuzzy-Predictive-Control repository is a comprehensive collection of Python Jupyter Notebooks that demonstrate the implementation of fuzzy modeling and predictive functional control techniques.

# Introduction

The fuzzy-predictive-control repository aims to support control engineers, researchers, and students in their pursuit of mastering advanced control techniques. The notebooks contain well-commented code examples, insightful visualizations, and comprehensive explanations to ensure a productive and educational learning experience. Whether you are a beginner or have some experience in control systems, these tutorials will help you enhance your control engineering skills and broaden your understanding of advanced control methodologies.

# Modelling basics

Notebook `basics.ipynb` serves as a foundational guide to basic modeling techniques for first-order dynamic systems. It covers essential concepts such as system identification, parameter estimation, and system response analysis. The notebook aims to build a strong understanding of the fundamental principles that underpin more sophisticated control strategies.

# Fuzzy predictive control

Focused on fuzzy (Takagi-Sugeno) modeling and predictive functional control, `fuzzy_control.ipynb` notebook presents a comprehensive approach to controlling the Helicrane system. It takes you through a series of well-structured steps, including:

## Static Characteristics: Analyzing the Helicrane's static behavior to gain insights into its characteristics.

We perform excitation of the system at different operational points on a scale and obtain process outputs.

| ![](/docs/img/static_characteristics_exp.png) |
|:--:| 
| Obtaining process static characteristics - experiment.|

Based on these outputs, we construct the static characteristic, illustrating the input-output relationship in a steady state
| ![](/docs/img/static_characteristics.png) |
|:--:| 
| Process static characteristics.|

## Fuzzy Clustering (Gustafson-Kessel): Implementing fuzzy clustering to partition data and generate fuzzy sets.

The Gustafson-Kessel method introduces a variable covariance matrix for each cluster and thus a concrete improves the detection of different cluster shapes in the dataset. In our case, we used it to identify local linear regions in the static characteristic of the process, in which we identified local linear models.

| ![](/docs/img/gk_clustering.png) |
|:--:| 
| Process static characteristics - clusters and covariance matrices.|

## Membership Functions: Visualizing and fine-tuning the membership functions for the fuzzy sets.

Based on the clusters detected in the static characteristic, we calculate membership functions describing the membership of the current state of the model to individual linear regions of the process.

Membership Functions are normalised Gaussian functions:

```math
\begin{align}
	\Phi_i(\mathbf{u}) = \frac{\mu_i(\mathbf{u})}{ \sum_{j=1}^{M} \mu_j(\mathbf{u})},
\end{align}
```
where:

```math
\begin{align}
	\mu_i(\mathbf{u}) = \prod_{j=1}^{p} e^{-\frac{u_j - c_{ij}}{2\gamma\sigma_{ij}}},
\end{align}
```
where:

- $M$: The number of LLM (Local Linear Model) instances.
- $p$: The number of parameters determining the memebership function.
- $u_j$: The value of the j-th element in the regressor matrix.
- $c_i$: The center of the i-th cluster.
- $\gamma$: The fuzziness coefficient used in the algorithm.
- $\sigma_{ij}$: The standard deviation.



| ![](/docs/img/membership_functions.png) |
|:--:| 
| Membership functions.|

## Local Linear Model Identification: Identifying local linear models to approximate the Helicrane's nonlinear behavior.

Estimating the local linear parameters of the model involves a linear optimization problem, assuming that the validity functions are known, which is the case in our study.

There are two approaches to parameter estimation:

 - Global estimation: Considers the model as a whole.
 - Local estimation: Considers only the local characteristics of the model.

In our case, we employed the local estimation approach due to its advantages over global estimation, such as reduced complexity and well-conditioned regressor matrices. The notable advantage of local estimation is its improved interpretability, which is crucial for identification purposes. However, this comes at the cost of increased bias.

The local estimation is performed separately for each local model, typically using methods like the least squares method.

The general transfer function sought in parametric identification is given as:

```math
\begin{align}
H(z) = z^{-d} \cdot \frac{(b_0z^n + b_1z^{n-1} + b_n)}  {(z^n + a_1z^{n-1} + a_n)}
\end{align}
```

To cover the entire operating range of the process, an pseudo-random binary signal (PRBS) was used as the excitation signal, with appropriate intervals for changing the amplitude (based on the dominant time constant of the process).

We construct the matrix $\mathbf{X}$ containing input process measurements and the output vector $\mathbf{y}$ containing output process measurements as follows:

```math
\begin{align}
	\mathbf{X} = 	\begin{bmatrix}
	 	x_1(1) & x_2(1) & \cdots & x_n(1) & 1 \\
     	x_1(2) & x_2(2) & \cdots & x_n(2) & 1 \\
    		\vdots & \vdots & \ddots & \vdots & \vdots \\
    		x_1(N) & x_2(N) & \cdots & x_n(N) & 1
    \end{bmatrix}\\
    \mathbf{y} = \begin{bmatrix} 
    		y(K) & y(K+1) & \cdots & y(N)
    \end{bmatrix}
\end{align}
```

The matrix $\mathbf{X}$ and the vector $\mathbf{y}$ in example of two delayed inputs and outputs are written:

```math
\begin{align}
\mathbf{X} &= 	\begin{bmatrix}
 	-y(2) & -y(1) & u(2) & u(1) \\
    -y(3) & -y(2) & u(3) & u(2) \\
    \vdots & \vdots & \vdots & \vdots \\
    -y(N-1) & -y(N-2) & u(N-1) & u(N-2)
\end{bmatrix}\\
\mathbf{y} &= \begin{bmatrix} 
    y(3) & y(4) & \cdots & y(N)
\end{bmatrix}
\end{align}
```

We calculate the model parameters $\mathbf{\Theta}$ using the following equation:

```math
\begin{align}
\mathbf{\Theta} = \left(\mathbf{X}^T \mathbf{X} \right)^{-1} \mathbf{X}^T \mathbf{y}.
\end{align}
```
Alternatively, parameter estimation with the method of weighted least squares (excitation with APRBS) is possible, where the weights in the diagonal matrix $\mathbf{Q}$ would be equal to the membership of the linear region. However, due to strong nonlinearity in the narrow final region, this approach did not perform well. While it may achieve smoother results and possibly reduce bias, it compromises interpretability.

## Fuzzy Model Simulation

 1. Fuzzy model can be simulated to observe its response to various inputs.
 2. One-Step Ahead Prediction method can be also used to predict the system's future behavior using the fuzzy model.


## Predictive Functional Controller (PFC): Designing a PFC for reference tracking and disturbance rejection in the Helicrane system.

In the case of controlling higher-order multivariable processes, it is more convenient to represent the model in the state-space form rather than using a transfer function. Thus, we focus on the following state-space representation:

```math
\begin{align}
	\mathbf{x}_m(k+1) = \mathbf{A}_m \mathbf{x}_m(k) + \mathbf{B}_m \mathbf{u}(k)\\
	\mathbf{y}(k) = \mathbf{C}_m \mathbf{x}_m(k) + \mathbf{D}_m \mathbf{u}(k)\nonumber
\end{align}
```

In real processes, where there is no direct influence of input signals on outputs, the term $\mathbf{D}_m \mathbf{u}(k)$ is omitted.


In our case, where a soft model composed of multiple locally linear models is used, we express the model in the Observable Canonical State Space (OCSS) form:

```math
\begin{align}
\mathbf{x}_m(k+1) = \tilde{\mathbf{A}}_m \mathbf{x}_m(k) + \tilde{\mathbf{B}}_m \mathbf{u}(k) + \tilde{\mathbf{R}}_m\\
\mathbf{y}(k) = \tilde{\mathbf{C}}_m \mathbf{x}_m(k)\nonumber,
\end{align}
```
where:

```math
\begin{align}
\tilde{\mathbf{A}}_m = \sum_{j=1}^{M} \beta_j \mathbf{A}_{m_j}\\
\tilde{\mathbf{B}}_m = \sum_{j=1}^{M} \beta_j \mathbf{B}_{m_j}\\
\tilde{\mathbf{C}}_m = \sum_{j=1}^{M} \beta_j \mathbf{C}_{m_j}\\
\tilde{\mathbf{R}}_m = \sum_{j=1}^{M} \beta_j \mathbf{R}_{m_j}.
\end{align}
```
Here, $\beta_j$ represents the weight, and $\mathbf{R}_{m_j}$ is the operating point of the j-th locally linear model.



```math
\begin{align}
Y(k) = -a_1 Y(k-1) - \ldots - a_m Y(k-m) + \nonumber\\
    b_1 U(k-1) + \ldots + b_m U(k-m) + \\
    (1 + a_1 + \ldots + a_m) \bar{Y} - (b_1 + \ldots + b_m) \bar{U}\nonumber
\end{align}
```
Steady-state  can be modelled as shown in the above equation, in the last row. This is denoted by $\tilde{\mathbf{R}}_m$.

During the implementation of PFC, a reference model trajectory is also modeled to represent the desired reference change (usually a first-order model). This reference model should have unity gain, meaning that the matrices $\mathbf{A}_r$, $\mathbf{B}_r$, and $\mathbf{C}_r$ must satisfy the condition:

```math
\begin{align}
\mathbf{C}_r\left(\mathbf{I} - \mathbf{A}_r\right)^{-1} \mathbf{B}_r = \mathbf{I}.
\end{align}
```

After the derivation, we obtain the simplified control law:


```math
\begin{align}
\mathbf{u}(k) = \mathbf{G}\left(\mathbf{w}(k)-\mathbf{y}_p(k)\right) + G_0^{-1}\mathbf{y}_m(k) - G_0^{-1} \mathbf{C}_m \mathbf{A}_m^H \mathbf{x}_m(k),
\end{align}
```
where:

```math
\begin{align}
\mathbf{G} = G_0^{-1}\left(\mathbf{I}- \mathbf{A}_r^H\right)
\end{align}
```
and:

```math
\begin{align}
G_0 = \mathbf{C}_m\left(\mathbf{A}_m^H - \mathbf{I}\right) \left(\mathbf{A}_m - \mathbf{I}\right)^{-1} \mathbf{B}_m.
\end{align}
```

Here, $\mathbf{w}$ represents the reference, and $H$ is the prediction horizon (during this horizon, the process output $\mathbf{y}_p(k)$ is expected to match the reference response $\mathbf{y}_r(k)$).

## Implementation

As mentioned earlier, it is necessary to first define the reference trajectory model, which, in our case, is a first-order model. The system matrix $\mathbf{A}_r$ is defined such that the time constant is 3-5 times smaller than the dominant time constant of the process. This parameter or system matrix $\mathbf{A}_r$ can be adjusted as desired to achieve the desired control objective.


The control is performed in the following sequence. First:

1. Select the reference signal.

2. Choose the dynamics of the reference generator and calculate matrices $\mathbf{A}_r$, $\mathbf{B}_r$, and $\mathbf{C}_r$.

3. Set initial states of the models and ensure the initial state of the process or adapt the model states to the current process state.
Then, in a loop:

4. Calculate the memberships $\mathbf{\beta}$ of LLM.

5. Calculate the output of the reference model and update the model states.

6. Calculate the output of the process model and update the model states.

7. Measure the process output.

8. Calculate the control variable for the next step.

9. Limit the calculated control variable according to actuator constraints.

10. Apply the control variable to the process.


The control strategy can be broadly divided into reference tracking and disturbance rejection operations.

### Reference tracking

During reference tracking, the goal is to achieve better tracking of the process output to the reference signal.

| ![](/docs/img/reference_tracking.png) |
|:--:| 
| Membership functions.|

### Disturbance rejection

While disturbance rejection, the aim is to minimize the effects of external disturbances at a constant reference.

| ![](/docs/img/disturbance_rejection.png) |
|:--:| 
| Membership functions.|

# Conclusions

Through this notebook, you will gain practical experience in applying advanced control strategies, with a specific focus on fuzzy logic and predictive control techniques. The presented techniques can be applied to other complex systems and real-world scenarios.


