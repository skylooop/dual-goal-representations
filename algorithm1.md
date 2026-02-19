\begin{aligned}
&\text { Algorithm } 1 \text { Offline Goal-Conditioned RL with Dual Goal Representations }\\
&\text { ▷ Dual goal representation learning }\\
&\text { Initialize state representation } \psi(s) \text {, goal representation } \varphi(g) \text {, Q function } Q(s, a, g)\\
&\text { while not converged do }\\
&\text { Sample batch }\left\{\left(s, a, s^{\prime}, g\right)^{(i)}\right\}_i \text { from } \mathcal{D}\\
&\text { - Train parameterized value function } f(\psi(s), \varphi(g)) \text { with goal-conditioned IQL }\\
&\text { Train } \psi, \varphi \text { by minimizing } \mathbb{E}\left[\ell_\kappa^2(f(\psi(s), \varphi(g))-\bar{Q}(s, a, g))\right]\\
&\text { Train } Q \text { by minimizing } \mathbb{E}\left[\left(Q(s, a, g)-r(s, g)-\gamma f\left(\psi\left(s^{\prime}\right), \varphi(g)\right)\right)^2\right]\\
&\text { Update } \bar{Q} \text { using exponential moving averaging }\\
&\text { - Downstream offline GCRL with dual goal representation (can be run in parallel with above) }\\
&\text { Initialize policy } \pi(a \mid s, \varphi(g))\\
&\text { (If necessary) initialize representation-conditioned value functions } V^{\mathrm{GCRL}}(s, \varphi(g)), Q^{\mathrm{GCRL}}(s, a, \varphi(g))\\
&\text { while not converged do }\\
&\text { Train } \pi, V^{\mathrm{GCRL}}, Q^{\mathrm{GCRL}} \text { with any offline GCRL algorithm (e.g., GCBC, GCIVL, CRL) }\\
&\text { return } \pi(a \mid s, \varphi(g))
\end{aligned}