import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stemgraphic
from scipy.special import comb  # For combinatorial calculations (see line 209)
from scipy.stats import hypergeom # For dhyper() equivalent (see line 225)
from scipy.stats import binom # For dbinom() equivalent (see line 264)
from scipy.stats import poisson # For dpois() equivalent (see line 304)
from scipy.stats import nbinom # For dnbinom() equivalent (see line 324)
from scipy.stats import geom # For dgeom() equivalent (see line 338)
from scipy.stats import norm # For pnorm() equivalent (see line 349)
from scipy.stats import probplot # For qqnorm() equivalent (see line 367)
from scipy.stats import lognorm # For plnorm() equivalent (see line 429)
from scipy.stats import expon # For pexp() equivalent (see line 438)
from scipy.stats import gamma # For pgamma() equivalent (see line 453)
from scipy.stats import weibull_min # For pweibull() equivalent (see line 462)

Temperatures = [953, 955, 948, 951, 957, 949, 954, 950, 959]


np.median(Temperatures)
avg = sum(Temperatures)/len(Temperatures)

#plot hist
plt.hist(Temperatures, bins='auto')
plt.show()

#stem plot
stemgraphic.stem_graphic(Temperatures, scale=1, leaf_order=True)
plt.show()

quarts = np.quantile(Temperatures, [.25, .50, .75])
print("Upper quartile " , quarts[2] , " and lower quartile " , quarts[0]) 

#box plot
plt.boxplot(Temperatures,vert=False)
plt.show()

#Q-Q plot
fig, ax = plt.subplots()
res = probplot(Temperatures, dist="norm") 
ax.scatter(res[0][0], res[0][1], label="Data Points")  # Scatter plot for octane data
ax.plot(res[0][0], res[1][1] + res[1][0] * res[0][0], color="red", label="Q-Q Line")  # Reference line

# Set labels and title
ax.set_title("Q-Q Plot with Data on X-axis")
ax.set_xlabel("Sample Quantiles (Semiconductor Data)")
ax.set_ylabel("Theoretical Quantiles (Normal)")

# Show legend and plot
ax.legend()
plt.show()

p = 0.15
k=3
probability = geom.pmf(k, p)
print(probability)

average=1/p
print(average)

#2.C
n = 50  # Number of trials
p = 0.15 # Probability of success
k=10 # number of success
prob=binom.pmf(k,n,p)
print(prob)

#3.a
N = 150  # Total components in the lot
K = 2   # Nonconforming components
n = 5   # Sample size
k = 0   # No nonconforming components in the sample

# Compute hypergeometric probability
probability = hypergeom.pmf(k, N, K, n)
print(probability)

#3.b
# Given values for binomial approximation
n = 150   # Sample size
p = 2/25  # Probability of selecting a nonconforming component
k = 0   # No nonconforming components

# Compute binomial probability
binomial_approx = binom.pmf(k, n, p)
print(binomial_approx)

#3c
# Given values for hypergeometric distribution
N = 150   # Total components in the lot
K = 2     # Nonconforming components
n = 5     # Sample size
k = 0     # No of nonconforming components

# hypergeometric probability
hypergeom_prob = hypergeom.pmf(k, N, K, n)

#  binomial probability approximation
p = K / N  # Probability of selecting a nonconforming component
binomial_prob = binom.pmf(k, n, p)

print("hypergeom_prob =", hypergeom_prob)
print("binomial_prob = ", binomial_prob)

#3d

N = 25  # Total lot size
K = 5   # Nonconforming components
target_prob = 0.95  # Desired rejection probability
k=0

# Find the smallest n such that P(reject) >= 0.95
for n in range(1, N + 1):
    P_accept = hypergeom.pmf(k, N, K, n)  # Probability of accepting the lot
    P_reject = 1 - P_accept  # Probability of rejecting the lot
    
    if P_reject >= target_prob:
        print("Minimum sample size required: ", n)
        break

#4
lambda_ = 0.1  


probability = 1 - poisson.pmf(0, lambda_)
print(probability)

#5
# Given values
mu = 5000
sigma = 50
probability = 0.005  # Lower 0.5% tail


lower_limit = norm.ppf(probability, loc=mu, scale=sigma)

print(lower_limit)

#6
lambda_ = 0.1  

# Define the exponential distribution scale parameter (1/lambda)
scale = 1 / lambda_  

# Probability of failure before 100 hours
P_fail_100 = expon.cdf(100, scale=scale)

# Probability of failure after 5 hours
P_survive_5 = 1 - expon.cdf(5, scale=scale)

print("probability that unit fails before 100 hours is " , P_fail_100*100,"%")
print("probability that unit fails after 5 hours is " , P_survive_5*100,"%")

#7
mu = 100
sigma = 2
LSL = 97
USL = 102

# Compute probabilities
P_LSL = norm.cdf(LSL, loc=mu, scale=sigma)
print(P_LSL)
P_USL = norm.cdf(USL, loc=mu, scale=sigma)
print(P_USL)
# Proportion within specifications
P_within_specs = P_USL - P_LSL
print(P_within_specs*100, "% is withinn specification")

#2nd part
mu_current = 100
sigma = 2
LSL = 97
USL = 102

# Compute probabilities for current mean
P_LSL_current = norm.cdf(LSL, loc=mu_current, scale=sigma) 
print("Probability of lower limit=", P_LSL_current*100)
P_USL_current = 1 - norm.cdf(USL, loc=mu_current, scale=sigma)
print("Probability of upper limit=", P_USL_current*100)

cost_scrap = 5  # Cost per scrapped unit
cost_rework = 1  # Cost per reworked unit

cost_current = (P_LSL_current * cost_rework) + (P_USL_current * cost_scrap)
print("current cost for one unit::", cost_current)

mu_new = 98
P_LSL_new = norm.cdf(LSL, loc=mu_new, scale=sigma)
P_USL_new = 1 - norm.cdf(USL, loc=mu_new, scale=sigma)

cost_new = (P_LSL_new * cost_rework) + (P_USL_new * cost_scrap)
print("new cost for one unit::", cost_new)

#7
n = 50  # Number of bets
p = 12 / 38  # Probability of winning
k = 11

# Compute probability using binomial distribution directly
probability_binom = 1 - binom.cdf(k, n, p)  # Calculating P(X > 11) which is P(X >= 12)
print("Probability of winning at least 12 times ", probability_binom*100)
