import numpy as np
from scipy import stats  # for Box-Cox Transformation
from mlxtend.preprocessing import minmax_scaling  # for min_max scaling

# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size=1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])

# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)
