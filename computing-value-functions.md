# Computing Value Functions

[In this post we will talk about how to directly compute a value function given knowledge of the MDP (and sufficient compute resources)]

[This is the difference between computing, estimating, and learning (something we will likely discuss in future posts). This post focuses on *computing*.]

[Let's be clear about our assumptions moving forward. In this post we will assume that we have direct access to the state-space. We will assume that states and actions are finite (thus countable). We also will assume we have direct access to quantities like $p(s' | s, a)$ and $p(r | s, a)$. All of these assumptions are not strictly necessary and, as we will see in future posts, much of the theory extends beyond these cases.]

## The transition probability matrix
The transition dynamics of an MDP can be generically expressed as the probabilities:

 * $\P{s' | s, a}$ - the probability of transitioning to state $s'$ given that I started in state $s$ and took action $a$.
 * $\P{r | s, a}$ - the probability of observing a reward of $r$ given that I started in state $s$ and took action $a$.
 * $\P{a | s}$ - the probability of taking action $a$ in state $s$. Often denoted as $\pi(a | s)$ in the RL literature (and called a "policy").

Using these fundamental quantities, we can construct everything else that we will need.

The first such constructed quantity is the probability that we transition to state $s'$ given we started in state $s$ when we behave according to policy $\pi$.
More succinctly, we want to know
$$
\begin{aligned}
    \P{s' | s}
        &= \sum_{a \in \mathcal{A}} \P{s' | s, a} \pi(a | s) && (1) \\
        &= \mathbb{P}_{a \sim \pi(\cdot | s)}\left( s' | s \right) && (2)
\end{aligned}
$$
where the summation is marginalizing over actions.
This is notationally a little clumsy.
We could define such a $\P{s' | s}$ for any given policy, but our notation does not state _which_ policy we sampled actions according to.
Often in statistics, we might denote this as Equation (2) to demonstrate under which probability law we consider our actions.
However, this notation is rather dense and can become visually burdensome later on.
So we are left to choose, a clumsy imprecise notation (1) or a dense, burdensome, but precise notation (2).

Equipped with the probability of transitioning from one state to the next, we can describe the transition probability matrix.
A transition matrix is a square matrix with one row and one column for every state.
You can interpret each row as being the current state and each column as being the next state, such that the $i$'th row and $j$'th column describes the probability of transitioning to state $j$ from state $i$.
That is, take your finger along the left side of the matrix and select a row---that is your starting state---then take another finger along the top of the matrix and select a column---that is your next state---bring your fingers together and the cell they land on describes the probability $P_{i,j} = \P{s_j | s_i}$, where we generally denote the probability matrix as $P \in [0,1]^{|\mathcal{S}| \times |\mathcal{S}|}$.
When we need to be clear which policy was used to generate the transition probabilities (recall Equation 1), then we denote this as $P_\pi$.

## Value functions
I will assume knowledge of the basic definitions of value functions.
For a quick primer, [this post](./bellman-consistency.md) has a formal definition.
Notice that
$$
\begin{aligned}
V(s)
    &\doteq \E{R + \gamma V(s') \; | \; s}
        && \anote{definition of value function} \\

    &= r(s) + \gamma \E{V(s') \; | \; s}
        && \anote{define $r(s)$ as the average reward in state $s$ following policy $\pi$} \\

    &= r(s) + \gamma \sum_a \pi(a | s) \sum_{s'} \P{s' | s, a} V(s')
        && \anote{decompose the expectation into sums} \\

    &= r(s) + \gamma \sum_{s'} \P{s' | s} V(s').
        && \anote{combine the marginalization over actions as in Equation (1)}
\end{aligned}
$$

## Extension to state-action values

[Notice this technical detail that I ignored: the invertibility of $(I - \gamma P_\pi)$, which is of course necessary for this whole discussion to hold. The next blog post will cover the proof of invertibility in detail. This is a little more involved so is relegated to a new post.]
