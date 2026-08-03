# Sigmoid
# the sigmoid graph looks like an S-curve.It starts close to 0 at z = -10 and reaches close to 1 at z = 10.The derivative is highest around z = 0 and decreases towards both sides.
# the output is between 0 and 1, so it is not zero-centered.

# Tanh
# the tanh graph is also S-shaped but passes through the origin.It changes from -1 to 1. The derivative is maximum at z = 0 and becomes very small near -10 and 10.
# Since it has both negative and positive values, it is zero-centered.

# ReLU
# the output is 0 for all negative values.from z = 0 onwards, the graph increases as a straight line.
# the derivative is 0 for negative values and 1 for positive values.
# it is not zero-centered because there are no negative outputs.

# Leaky ReLU
# the graph is similar to ReLU.for negative values, the output decreases slowly instead of becoming 0.
# for positive values, it behaves exactly like ReLU.
# the derivative is small (0.05) for negative values and 1 for positive values.

# Softmax
# the outputs are probabilities.
# all output values are between 0 and 1.
# the sum of all probabilities is 1.