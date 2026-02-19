1 Dual Advantage Fields
1.1 Research Proposal
Describe your idea in a few sentences.
We propose an algorithm for efficient policy extraction from dual goal representation. Starting from
the dual goal representations in [PML25], where the goal-conditioned potential is parameterized by
an inner product Vθ(s,g) ≡f(ψθ(s),ϕθ(g)) = ψθ(s)⊤ϕθ(g), we show that the goal embedding ϕθ(g) is
exactly the gradient of the learned potential V with respect to the state representation coordinates:
∇ψVθ(s,g) = ϕθ(g). Using this, we derive a dual advantage estimator that enough for policy iteration.
1.2 Motivation
Describe the problem you want to solve. Is it important?
Dual goal representations encode a goal through its temporal relations to all states and provide appealing
theoretical properties (e.g., sufficiency and noise invariance). However, the practical algorithm trains
a structured inner-product temporal-distance/value proxy for representation learning and then
commonly relies on a separate downstream offline GCRL loop for policy extraction. This
two-stage design reflects a conceptual mismatch: a temporal-distance or value field is a global scalar
map, whereas policy improvement fundamentally depends on local action comparisons. Moreover,
[PML25] noted that directly extracting a policy from the inner product V failed because the
value surface was too rigid. Therefore, connecting dual representations to advantage computation is
important for (i) more direct policy improvement, (ii) potentially simpler pipelines (less reliance on a
second value learner), and (iii) better utilization of the structure already imposed by dual spaces.
1.3 Related Work
What is available on this topic? Papers first, then implementations and everything else. For
each paper, a short summary to clarify its focus, results, reliability, and reproducibility.
Dual goal representations. [PML25] define
the dual goal representation ϕ∨(g) as the func-
tional mapping each state to the optimal temporal
distance to g, and show theoretical sufficiency and
noise invariance. Practically, they approximate
the temporal-distance function via a parameteri-
zationd∗(s,g) = f(ψ(s),ϕ(g)) withaninnerprod-
uct f(ψ,ϕ) = ψ⊤ϕ, and use goal-conditioned IQL
to learn this proxy. Their Algorithm 1 then
trains a downstream goal-conditioned pol-
icy π(a |s,ϕ(g)) with an arbitrary offline
GCRL method. They further report that
Figure 1: Illustration from the original paper on
dual representations
directly extracting a policy from the struc-
tured inner-product function can degrade
performance, motivating the separate downstream loop.
Policy improvement using advantages. [DS95] argue that policy iteration should only require
relative measures of action utility (advantages), not absolute values. In their continuous deterministic
setting, they define an advantage Aw(x,u) = r(x,u) + f(x,u) ·∇xVw(x) and improve policies by
maximizing Aw. They emphasize that learning the relevant derivatives can be underconstrained and
introduce conservative-field (curl-free) structure as an additional constraint. This peer-reviewed NeurIPS
paper is conceptually reliable; translating its differential constraints to modern deep, discrete-time
offline RL requires careful design.
2
1.4 Idea
How do you propose to solve the problem? What is new about this solution? Give a fairly
detailed description.
∂ψj
∂ψj
Step 1: Fix the representation class We adopt [PML25]’s practical instantiation:
Vθ(s,g) ≜ f(ψθ(s),ϕθ(g)) = ψθ(s)⊤ϕθ(g), (1)
where ψθ : S→RN and ϕθ : S→RN are state and goal heads, and Vθ is trained as a goal-conditioned
temporal-distance/value proxy using an offline value-learning algorithm (goal-conditioned IQL).
Step 2: Prove (in the model class) that ∇ψVθ(s,g) = ϕθ(g). The key identity follows from basic
multivariate calculus and the linearity of the chosen aggregator. Fix any goal g. Define Vθ(·,g) as a
function of the representation vector ψ∈RN:
Vθ(ψ; g) ≜ ψ⊤ϕθ(g) =
N
i=1
ψiϕθ,i(g). (2)
For each coordinate j ∈{1,...,N}, the partial derivative is
∂Vθ(ψ; g)
∂
=
N
i=1
ψiϕθ,i(g) = ϕθ,j(g), (3)
because ϕθ(g) does not depend on ψ, and ∂ψi/∂ψj= I[i= j]. Collecting these partial derivatives yields
the gradient vector:
∇ψVθ(ψ; g) = ∂Vθ
∂ψ1
⊤
∂Vθ
∂ψN
= [ϕθ,1(g),...,ϕθ,N(g)]⊤
= ϕθ(g). (4)
Since Vθ(s,g) = Vθ(ψθ(s); g), the identity ∇ψVθ(s,g) = ϕθ(g) holds everywhere in representation space
by construction of the inner-product model.
Intuition: This identity provides the critical geometric bridge between the two papers. It reveals that
[PML25]’s goal representation φ(g) is not merely a latent embedding; it is precisely the gradient of
the value function in latent space. By learning the dual representation, we are implicitly learning the
derivative ∇V required by [DS95].
Step 3: Derive a dual advantage formula. [DS95] express advantages as reward plus an inner
product between a dynamics direction and a value gradient. In discrete-time goal-conditioned RL,
define the standard advantage:
,...,
A(s,a,g) ≜ Q(s,a,g)−V(s,g). (5)
Using [PML25]’s TD-style target for the learned Qhead in Algorithm 1, Q(s,a,g) ≈r(s,g) + γV(s′,g),
and substituting V(s,g) = ψ(s)⊤ϕ(g), we obtain
A(s,a,g) ≈r(s,g) + γψ(s′)⊤ϕ(g)−ψ(s)⊤ϕ(g)
= r(s,g) + γψ(s′)−ψ(s) ⊤ϕ(g). (6)
By (4), ϕ(g) is the gradient of the potential in ψ-coordinates, and γψ(s′)−ψ(s) is the action-conditional
direction of change in those coordinates. Equation (6) is therefore the discrete, representation-space
analogue of [DS95]’s advantage formula.
Intuition: This formula represents the exact discrete analogue of [DS95]’s advantage equation A=
r+ ˙
x·∇V. Here, γψ(s′)−ψ(s) represents the “change in state” vector, and φ(g) represents the
gradient. We have successfully substituted the abstract gradient term from [DS95] with the learned
goal representation from [PML25], unifying the two approaches.
3
Step 4: Practical estimator and policy improvement. We can say [PML25] provide the map
(a potential), while the dual advantage provides the compass (directional action scores). So, to avoid
relying on a sampled s′at decision time, we learn an action-effect head uξ(s,a) ≈E[γψ(s′)−ψ(s) |s,a]
by regression on offline transitions. We then define
A(s,a,g) ≜ r(s,g) + uξ(s,a)⊤ϕθ(g), (7)
and perform policy improvement by selecting actions that maximize A, consistent with the advantage-
drivenimprovementprinciplein[DS95]. Optionally,weregularizeuξ withDayan-inspiredcycle/conservative-
field constraints to improve generalization beyond actions heavily represented in the dataset.
Justification: [PML25] noted that directly extracting a policy from the inner product V failed because
the value surface was too rigid. Our approach solves this by using φ(g) as a local compass (gradient)
rather than a global map. Instead of asking “which state has the highest inner product?” (rigid), we
ask “which action moves me in the direction of φ(g)?” (flexible). This aligns the action selection with
the local advantage landscape, rather than the global value approximation.
1.5 Discussion
Why would it work? What conclusion and new intuition we gets. Bonus: why it may not work?
What we should pay attention to?
The identity ∇ψVθ(s,g) = ϕθ(g) shows that Park’s goal embedding is not merely a descriptor of a
target state; it is the dual object that evaluates local directions in representation space. Consequently,
combining ϕ(g) with an action-effect vector yields an advantage signal that is inherently local and
action-comparative, precisely aligned with advantages suffice for policy iteration.
Potential failure modes include: (i) representation collapse or misalignment, where ψ does not parame-
terize controllable directions well; (ii) stochastic or multi-modal transitions, where a mean action-effect
uξ may be insufficient; and (iii) offline extrapolation, where maximizing A exploits approximation
error. [DS95]’s analysis suggests that additional structural constraints (e.g., conservative-field/cycle
conditions) can be important when the directional information is underdetermined, motivating explicit
regularization of the learned action-effect field.
1.6 Experiments A
How to quickly test an idea to see if it works (conventionally, in a couple weeks) to decide if it’s
worth exploring further.
Implement a minimal extension of [PML25]’s Algorithm 1: learn (ψ,ϕ) with the same inner-product
value proxy and then train uξ(s,a) by regressing to γψ(s′)−ψ(s) on the offline dataset. Compare: (i)
Park dual representations + downstream GCRL (baseline) versus (ii) Dual-Advantage using A(s,a,g) =
r(s,g)+uξ(s,a)⊤ϕ(g) and an advantage-weighted actor update. Evaluate on a small subset of OGBench
tasks, reporting success rates and variance across seeds. Key diagnostic: correlation between A and
empirical short-horizon returns for held-out (s,g) pairs. Ablate the cycle/conservative regularizer
inspired by [DS95].
1.7 Experiments B
If Experiment A is successful, what are the next steps?
Scale evaluation to the full benchmark suite used by [PML25] and test whether dual advantages can
reduce dependence on a separate downstream value learner without degrading performance. Extend to
pixel-based settings, where [PML25] note late-fusion limitations for representation-conditioned policies,
and assess whether a state-aware variant of the dual advantage (e.g., learning uξ from joint state-goal
features) addresses these issues.