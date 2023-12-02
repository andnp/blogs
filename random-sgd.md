# A reminder that Stochastic Gradient Descent is a random process.

I recently got into a discussion with a group of machine learning enthusiasts who seemed to be under the misconception that stochastic gradient descent was not random.
And yeah it seems a bit obvious, the word "stochastic" is synonymous with the word "random" so clearly "random" gradient descent is random; it isn't clear that this misconception directly requires an entire article dedicated to dispelling it.
However, there were some underlying subtleties to this belief that I realized were quite easy to fall victim to.

Taking a step back, consider a few true statements that might feel a bit misleading.

1. SGD updates in a random direction on every step.
2. SGD moves randomly through parameter (weight) space.
3. For general non-linear functions, SGD converges to a random point.

Initially these seem like they must be untrue.
How can SGD move in a random direction? We've been told since Machine Learning 101 that SGD was guaranteed to find a minimum!
Not only this, but we can also write out the exact procedure that SGD uses to find its minimum.
This procedure involves taking gradients of a known function, applying some known data from a dataset, then moving in a very predictable direction.
This seems so different from random search.

## SGD uses a deterministic update rule
