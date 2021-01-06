# Can we ignore the double-sampling issue in RL?

For anyone who has read the Sutton and Barto reinforcement learning textbook--specifically chapter 11--the problem of double sampling has come up and was neatly avoided by a simple trick using Gradient Temporal Difference learning methods.
However, it may not be clear why the trick used by GTD methods manages to avoid double sampling, and worse, it seems that trick only works for the linear function approximation case.
Likely, you've heard about the divergence issues with Q-learning and the deadly triad, but have happily applied Q-learning to your problem setting anyways because it works and is easy to extend to nonlinear function approximation (e.g. Neural Networks).

It turns out that extending GTD methods to nonlinear function approximation is also not too difficult, in terms of implementation, but maintaining the nice theoretical guarantees (i.e. guaranteed convergence) of GTD methods requires jumping through some mathematical hoops.
This post will be more focused on defining the core theoretical aspect of extending GTD methods to nonlinear function approximation.
A follow-up post will demonstrate the simplicity of the implementation and will motivate why it might be a good idea practically to drop semi-gradient based methods in favor of gradient based methods.

## Double-sampling
Before discussing how to avoid the issue of double sampling, it might be good to first recall what the problem is.
At a high level, double sampling occurs when we would like to estimate a product of expected values using sample averages.
Consider the case when we have two random variables, say the height of a person $H$ and their shoe size $S$.
Suppose we would like to know the average ratio between a person's height and their show size,
$$
\E{\frac{H}{S}} = \E{H}\E{S^{-1}} + \text{Cov}(H, S^{-1}).
$$

We go around and ask a lot of people what is their height and their shoe size and we notice a clear relationship between these variables.
As height increases, generally so does shoe size; that is, they are positively correlated.
For each person whose information we gathered, we compute a sample ratio of that individual's height and shoe size.
We compute a sample average (add up the sample ratios and divide by the number of people we've queried) and report the average ratio.

A friend of ours requests to borrow our dataset and checks our work.
However, they first compute the sample average height then the sample average shoe size (inverse) and multiply these sample averages together.
Our answers are both similar, but different.
Our estimate using the sample ratios includes information about the correlation between height and shoe size, while our friend's estimate ignores this information.
Who is correct depends on the scenario (though in this case, probably we are correct because we want to include information about individual differences in our estimate).

This same scenario occurs when computing the gradient of an objective function in RL.
If we take the Mean-Squared Bellman Error to be our objective of choice, then compute its gradient, we get
$$
\begin{aligned}
    \nabla_w \overline{\text{MSBE}}(w)
        &= \nabla_w \E{R + \gamma V_w' - V_w}^2 \\
        &= 2 \E{R + \gamma V'_w - V_w} \nabla_w \E{R + \gamma V'_w - V_w}
            && \anote{chain rule} \\
        &= 2 \E{R + \gamma V'_w - V_w} \E{\gamma \nabla_w V'_w - \nabla_w V_w}
            && \anote{linearity of expectation} \\
        &= 2 \E{\delta} \E{\nabla_w \delta}.
            && \anote{more succinct}
\end{aligned}
$$
Notice here our problem.
To compute this gradient, we need to compute the product of two expectations where $\delta$ is our "height" and $\nabla_w \delta$ is our "shoe size".

Clearly these random variables are correlated; as the magnitude of our error ($\delta$) increases, then so too does the magnitude of our gradient ($\nabla_w \delta$).
Not only the magnitudes are correlated, but likely the direction of the gradient is correlated with the magnitude of error (perhaps high error states also share the same set of features).
Unlike in the "height-shoe size" example, it is less straightforward to compute the product of these expectations.
Both expectations depend on the quantity $V_w(S')$ where $S'$ is the next-state.
To compute the product of expectations requires that we have two independent samples of $S'$ following a particular state $S$.
In the continuous or high-dimensional state settings, we are very unlikely to see multiple transitions from a state $S$ and even if we did, we likely wouldn't have enough memory to store all of these transitions.
Further, even if we did get multiple transitions from every state and we had enough memory to store all of these, we wouldn't be able to make a single update to our algorithm until the second time we've visited each state, significantly decreasing our sample complexity and delaying our time to first start learning.

## Can we ignore it?

Presumably the most simple way to avoid double sampling is to ignore that it even exists.
We can happily draw samples that include the individual correlations for each state, so what happens if we simply include those correlations?
We end up with a sample gradient that looks benign enough
$$
2 (r + \gamma V'_w - V_w) (\gamma \nabla_w V'_w - \nabla_w V_w).
$$
Each of these values can be directly sampled from the environment during online learning and require constant time and memory to evaluate.
And unlike the TD update, this sample update even is the gradient of a known objective function.

If instead of minimizing the MSBE, we chose to move the "square" to the inside of the expectation we get the Mean-squared Temporal Difference Error function
$$
\begin{aligned}
    \nabla_w \overline{\text{MSTDE}}(w)
        &= \nabla_w \E{(R + \gamma V'_w - V_w)^2} \\
        &= \E{\nabla_w (R + \gamma V'_w - V_w)^2}
            && \anote{by linearity} \\
        &= 2 \E{(R + \gamma V'_w - V_w) \nabla_w (R + \gamma V'_w - V_w)}
            && \anote{chain rule} \\
        &= 2 \E{(R + \gamma V'_w - V_w)(\gamma \nabla_w V'_w - \nabla_w V_w)} \\
        &= 2 \E{\delta \ \nabla_w \delta}.
            && \anote{make succinct}
\end{aligned}
$$

So we know we are taking the gradient of a known objective, meaning a simple stochastic gradient descent procedure can trivially yield convergence guarantees.
We can stochastically sample estimates of this gradient cheaply.
What's wrong with this approach?

Recall a fundamental theorem in reinforcement learning:
> The optimal value function $v_\pi$ uniquely solves the Bellman Equation $\text{B}v_\pi = v_\pi$ for Bellman Operator $\text{B}V = R + \gamma V'$.

Using this theorem, a natural idea would be to try and minimize the distance between $\text{B}V$ and our estimate $V$, knowing that if this distance is zero then we must have found the optimal value function $v_\pi$.
One distance metric we could consider is the mean-squared error, where we compute squared distances between the Bellman step $\text{B}V$ and the estimate $V$ for every state, then average these distances together.
$$
\overline{\text{MSBE}} = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \text{dist}\(\text{B}V(s), V(s) \)
$$

Minimizing this objective to zero guarantees that we will find the optimal value function.
But what happens if we minimize the MSTDE to zero instead, as we will try to do using our gradient procedure that ignores double sampling?
$$
\E{\delta} \E{\nabla_w \delta} = \E{\delta \nabla_w \delta} - \text{Cov}(\delta, \nabla_w \delta)
$$
which implies
$$
\underbrace{\text{dist}\( \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \text{B}V(s), \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} V(s) \)}_{\overline{\text{TDE}}}
=
\underbrace{\frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \text{dist}\(\text{B}V(s), V(s) \)}_{\overline{\text{BE}}}
+ \underbrace{\frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \text{Cov}(\text{B}V(s), V(s))}_{\text{some leftovers}}.
$$

This is a problem.
Minimizing the MSTDE includes minimizing some residual leftover term due to the covariance in our errors.
Reaching the zero of the MSTDE would guarantee that we also reach the zero of the MSBE (because both MSBE and the leftovers are positive); however it is often impossible to reach the zero of the MSTDE.
To intuitively see why, consider that the leftover covariance term depends on the variance of the rewards on each state; a variance that the learning algorithm has no agency over (the agent cannot control this variance by changing its weights $w$).
If we can't reach the zero of the MSTDE, then that leaves open the question of where we _do_ end up instead, but it is not guaranteed to be a point where the MSBE is also zero.
Without this, we can guarantee convergence (and thus that the learning process will be well behaved) but we cannot guarantee where we will converge to (and thus the final asymptotic error in the solution).
In practice, this solution turns out to often be pretty poor.

[TODO: show some results investigating the fixed-point of the MSTDE objective as well as learning curves of the residual gradient algorithm]
