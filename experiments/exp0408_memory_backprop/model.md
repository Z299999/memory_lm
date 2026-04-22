# An Age-Structured Formulation of Compressed Memory Dynamics

## Continuous Formulation

We begin with the classical age-structured transport model. Let $m(t,a)$ denote the memory density at time $t \ge 0$ and age $a \in [0, a_{\max}]$. The standard PDE formulation is:

$$\partial_t m(t,a) + \partial_a m(t,a) = -\mu(a)\, m(t,a), \qquad t>0,\quad 0<a<a_{\max}$$

$$m(0,a) = m_0(a), \qquad 0 \le a \le a_{\max}$$

$$m(t,0) = u(t), \qquad t \ge 0$$

Here $\mu(a) \ge 0$ is the age-dependent forgetting rate, $m_0$ is the initial memory profile, and $u(t)$ is the newly written memory at age $a=0$.

## Discrete Formulation

Motivated by the discrete token nature of language models, we pass to a discrete-time, discrete-age formulation. Let:

- $x_t$: user input at step $t$
- $y_t$: memory text read from the memory bank
- $r_t$: model response
- $m_t^{(a)}$: memory chunk of age $a \in \{0, 1, \dots, A_{\max}\}$ at step $t \in \mathbb{N}$

The compressed memory dynamics are:

$$y_t = \mathcal{R}_L\!\left(m_t^{(0)}, m_t^{(1)}, \dots, m_t^{(A_{\max})}\right)$$

$$r_t = G_\theta(x_t, y_t)$$

$$m_{t+1}^{(0)} = r_t$$

$$m_{t+1}^{(a+1)} = F_a\!\left(m_t^{(a)}\right), \qquad a = 0, 1, \dots, A_{\max}-1$$

$$|F_a(m)| \le \rho(a)\,|m|, \qquad 0 < \rho(a) \le 1, \qquad \rho(a+1) \le \rho(a)$$

where $\mathcal{R}_L$ is the readout operator that concatenates and truncates the age-ordered memory into a feasible context of length at most $L$, $G_\theta$ is the language model with parameters $\theta$, and $F_a$ is the age-dependent compression operator. The inequality enforces that older memory is no longer than younger memory after compression.

## Forgetting Curves

To model age-dependent compression in the spirit of the Ebbinghaus forgetting law, one may take the exponential form:

$$\rho(a) = e^{-\lambda a}, \qquad \lambda > 0$$

or the hyperbolic form:

$$\rho(a) = \frac{1}{(1+\beta a)^p}, \qquad \beta > 0,\quad p > 0$$

Both satisfy $\rho(0) = 1$ and yield a monotone decay of memory length with age, thereby controlling the total size of the readable memory text.

## Baby Case ($A_{\max} = 0$)

With $A_{\max} = 0$, there is only one memory slot $m_t^{(0)}$:

$$y_t = m_t^{(0)} = r_{t-1}$$

$$m_{t+1}^{(0)} = r_t$$

The model sees only the previous response as memory, and each new response completely replaces it.
