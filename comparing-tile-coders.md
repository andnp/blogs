# Comparing Tile-coding Implementations

This post discusses a direct approach to implementing a tile-coding representation, comparing both the computational and learning performance to the popular Tiles3 implementation.
I point out a small bug in Tiles3 and show how in a particular setting this bug can have a large impact on early learning results.

## Tile-coding

I won't go into a deep tile-coding primer, a fantastic introduction to tile-coding can be found in [Sutton and Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html).
But I _will_ discuss a few important attributes of tile-coding that will help shape the discussion of the difference between implementations.

A tile-coder is a fixed representation generating function.
I say _fixed_ because the function does not change over time unlike---for instance---a neural network.
The tile-coder maps a state from the set of states to a vector of features, or mathematically:
$$
\Phi : \mathcal{S} \to \mathcal{X}, \quad \mathcal{S} \in \R^m, \mathcal{X} \in \R^n.
$$
Which states that the tile-coder $\Phi$ maps from state (represented as an $m$ vector) to a feature vector with $n$ dimensions.

We can then take the generated nonlinear representation and feed that to our favorite reinforcement learning algorithm.
Traditionally, tile-coded representations have been paired with linear function approximation algorithms (linear in the representation, but nonlinear in the states); however, this need not be the case.
For instance, a [recent AAAI paper](https://arxiv.org/abs/1805.07476) demonstrates that feeding a tile-coded representation into a neural network helps decrease the impact of catastrophic interference.
Another [recent work](https://arxiv.org/abs/1911.08068) illustrates the effectiveness of using tile-coding as an activation function on each layer of a neural network, leading towards sparse neural net representations.
For simplicity, this post will focus on the linear function approximation case.

A tile-coder contains a set of tilings.
To visualize a single tiling in a one-dimensional state setting, one can imagine clumping nearby states together into one "macro-state" or "tile".
In the 1D case, this is often called "state-aggregation".
In the two-dimensional setting, imagine taking a grid and laying it overtop the statespace, discretizing states by again clumping together nearby states into a rectangle.
A tile-coder takes many such tilings, queries which tile a state belongs to for every tiling, then returns an N-hot representation where exactly one tile from every tiling is active.

From this intuitive definition, it's not too difficult to attempt an implementation of a single tiling.
The function needs to know how many dimensions we are laying our "grid" over, how many times we are slicing each dimension (e.g. in the 1D case how many macro states we have), and the coordinates of the tile we seek.
The function should return an index indicating which tile the coordinates land in.
```python
def getTileIndex(dimensions: int, tiles_per_dim: int, coords: List[float]) -> int:
    # the index of the tile that coords is in
    ind = 0

    # length of each tile in each dimension
    # if there are 2 tiles, then each is length 1/2
    # if there are 3 tiles, then each is length 1/3
    # ...
    tile_length = 1 / tiles_per_dim

    # the total number of tiles in a 1D space
    # is exactly the number of requested tiles.
    # in 2D space, we have a square so we square the number of tiles
    # in 3D space, we have a cube so we cube the number of tiles
    # ...
    total_tiles = tiles_per_dim ** dimensions

    for dim in range(dimensions):
        # in this particular dimension, find out how
        # far from 0 we are and how close to 1
        ind += coords[d] // tile_length

        # then offset that index by the number of tiles
        # in all dimensions before this one
        ind *= tiles_per_dim**d

    # make sure the tile index does not go outside the
    # range of the total number of tiles in this tiling
    # this causes the index to wrap around in a hyper-sphere
    # when the inputs are not between [0, 1]
    return ind % total_tiles
```

To consider multiple tilings only requires a very slight modification.
We need to perturb the coordinates for each tiling so that the grids are all offset from each.
Without this, each grid would lie exactly on top of each other and return the same value; which of course does little to help our generalization or discrimination properties!

```python
def getTilingsIndices(dimensions: int, tiles: int, tilings: int, coords: List[float]) -> List[int]:
    tiles_per_tiling = tiles**dimensions
    tile_length = 1 / tiles

    indices = np.empty(tilings)
    for tiling in range(tilings):
        # offset each tiling by a fixed percent of a tile-length
        # first tiling is not offset
        # second is offset by 1 / tilings percent
        # third is offset by 2 / tilings percent
        # ...
        offset = tile_length * tiling / tilings

        # because this wraps around when the inputs are
        # bigger than 1, this is a safe operation
        ind = getTileIndex(dimensions, tiles, coords + offset)

        # store the index, but first offset it by the number
        # of tiles in all tilings before us
        indices[tiling] = tiles_per_tiling * tiling

    return indices
```

And that's the whole tile-coder!
There are various other ways to construct a similar process that results in a series of offset one-hot encoders, but this is the most literal translation from the RL textbook.
The most commonly used implementation ([written by Rich](http://www.incompleteideas.net/tiles/tiles3.html)) is built based on hashing tile coordinates and is a rather indirect implementation.
I'll call my implementation the "direct" implementation and I will call Rich's implementation Tiles3.

## Comparing implementations

I was curious how my tile-coder implementation compared to Tiles3, both in terms of compute performance and agent learning performance, so I implemented the canonical Mountain Car domain with a SARSA agent.
I performed a small sweep across tile-coder parameters---number of tiles and number of tilings---then swept for a good stepsize for each tile-coder parameter combination.
I ran each combination 2000 times for each implementation and compared the online average return per timestep between both implementations.

![[Learning Curve Comparison]](./images/tc-sarsa-mc-lc.png)

On the y-axis, I show the return obtained for a given timestep, averaged over 2000 runs.
On the x-axis, is the number of steps each algorithm was run for (each X-label is steps/100 due to subsampling).
The shaded region corresponds to a 95% confidence interval, though with 2000 samples each this region will be fairly small.
For this particular problem, it appears that the tile-coders perform very similarly when a larger number of tiles is used, but the Tiles3 implementation under-performs the more direct implementation when using a small number of tiles.

There's actually a good reason for this.
The Tiles3 implementation is not careful about floating point error and maintaining strict boundaries between tilings.
Notice in the direct implementation, when we decide which tile a coordinate belongs to, we wrap the tile index around so that it can never be greater than the total number of tiles.
If we **did not** do this then each tiling would be allowed to "reach out" into other tilings, making it possible that one tiling would have zero active tiles while another tiling would have two active tiles.

This breach of contract has fairly minimal implications when there are many tiles and tilings.
As the number of total tiles in the entire tile-coder grows, then responsibility of each tile to be accurate goes down.
When there are very few tiles, then each shares considerably more responsibility.
The total number of features in a tile-coder can be computed by
$$
    \text{features} = \text{tilings } \times \text{ tiles}^\text{dims}.
$$

So in this Mountain Car example---which has two dimensions---the responsibility of each feature shrinks quadratically with the number of tiles per dimension and proportionally to the number of tilings.

Learning curves give a nice representation of the magnitude of performance for an agent, but they make it rather difficult to carefully compare two different agents.
Instead, it can often be much more enlightening to investigate a baseline difference curve which demonstrates the difference in performance between two algorithms over time.

![[Difference Curve]](./images/tc-sarsa-mc-diff.png)

Here we can make more statistically significant claims about the empirical differences between algorithms as well.
If the dashed black line at zero is contained by the shaded region, then there is no detectable difference between algorithms.
When The difference is positive then the direct tile-coder implementation is better, and when negative the Tiles3 implementation is better.
Again the x-axis shows the number of training steps divided by 100.

Notice here that the differences between the small representations (e.g. blue and orange) is much more pronounced and is statistically significantly different.
The magnitude of difference is much higher for early learning and decays towards zero over time, though notice the shaded region never contains zero.
For the larger representations (red and green), there is no detectable difference between the tile-coder implementations and the variance between each is quite large in early learning.
This suggests that for several of the 2000 runs, my direct implementation outperformed Tiles3; while for several other runs Tiles3 outperformed mine.

It might also be interesting to compare the distribution over the 2000 generated learning curves, to see if one tile-coder implementation yields a different distribution than the other.
From the learning curves, we might guess that each is low-variance with a well-behaved mean because the curves are so smooth and the shaded-region so small.
From the difference curves, we might expect something entirely different due to the jagged mean line and large shaded regions.

![[Performance Distributions]](./images/tc-sarsa-mc-dist.png)

Here we see the average return across all episodes on the x-axis and the empirical probability of attaining that level of performance on the y-axis.
While most of the distributions appear approximately normal, it does seem that each has a bit of skew towards the left.
This implies that a small number of runs would occasionally fail and obtain a very small average return, which is perhaps not too surprising.

As a closing remark, I had mentioned comparing computational efficiency of each implementation.
I found that both implementations are quite similar in this regard natively.
The direct implementation can be implemented with stronger type-safe data structures (i.e. numpy arrays), which allows considerable speedups using preprocessing libraries like [Numba](https://numba.pydata.org/).
The hash function and table lookup implementation of Tiles3 precludes a naive inclusion of Numba, though likely it could be adapted to attain similar performance.
In profiling each of my 2000 runs for both implementations, I found that the direct implementation (with Numba) took approximately 3.6s per run for the large representations where Tiles3 took 6.5s per run.
For small representations, the gap was less pronounced at 3.0s per run for mine and 4.7s for Tiles3 largely due to Python's overhead.

All of the code used to generate these results, as well as some additional exploration, is available on [Github](https://github.com/andnp/ComparingTileCoders).
Additionally, the efficient and unit-tested version of the direct tile-coder can be installed as a python library from Github [here](https://github.com/andnp/PyFixedReps).
