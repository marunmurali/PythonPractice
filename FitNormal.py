import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from pandas import * #for csv


# Generate some data for this demonstration.
dataSource = read_csv("/home/arun/flaptter_ws/Data/Noise/noise.csv")
data = np.array(dataSource['noise'].tolist())

# Fit a normal distribution to the data:
mu, std = norm.fit(data)
print(mu)
print(std)
# Plot the histogram.
plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()