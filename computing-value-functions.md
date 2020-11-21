# Computing Value Functions

[In this post we will talk about how to directly compute a value function given knowledge of the MDP (and sufficient compute resources)]

[This is the difference between computing, estimating, and learning (something we will likely discuss in future posts). This post focuses on *computing*.]

[Let's be clear about our assumptions moving forward. In this post we will assume that we have direct access to the state-space. We will assume that states and actions are finite (thus countable). We also will assume we have direct access to quantities like $p(s' | s, a)$ and $p(r | s, a)$. All of these assumptions are not strictly necessary and, as we will see in future posts, much of the theory extends beyond these cases.]

## The transition probability matrix
The transition dynamics of an MDP can be generically expressed as the probabilities:

 * $\P{s' | s, a}$ - the probability of transitioning to state $s'$ given that I started in state $s$ and took action $a$.
 * $\P{r | s, a}$ - the probability of observing a reward of $r$ given that I started in state $s$ and took action $a$.
 * $\P{a | s}$ - the probability of taking action $a$ in state $s$. Often denoted as $\pi(a | s)$ in the RL literature (and called a "policy").

Using these fundamental quantities, we can construct everything else that we will need in this post.

The first such probability that we will need to know is the probability that we transition to state $s'$ given we started in state $s$ when we behave according to policy $\pi$.
More succinctly, we want to know
$$
    \P{s' | s, a \sim \pi(\cdot | s)} = \sum_{a \in \mathcal{A}} \P{s' | s, a} \pi(a | s)
$$
where the summation is [marginalizing](../TODO.md) over actions.
The notation on the left-side of the equal sign states "the probability of observing a particular next state $s'$ given that we are in state $s$ and that actions are selected according to policy $\pi$ at that state."

Equipped with the probability of transitioning from one state to the next, we can describe the transition probability matrix.
A transition matrix is a square matrix with one row and one column for every state.
You can interpret each row as being the current state and each column as being the next state, such that the $i$'th row and $j$'th column describes the probability of transitioning to state $j$ from state $i$.



## Value functions

## Extension to state-action values

[Notice this technical detail that I ignored: the invertibility of $(I - \gamma P_\pi)$, which is of course necessary for this whole discussion to hold. The next blog post will cover the proof of invertibility in detail. This is a little more involved so is relegated to a new post.]
