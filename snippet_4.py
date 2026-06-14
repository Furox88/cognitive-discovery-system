from cds.probability import gaussian_pdf, binomial_pmf, poisson_pmf

# Gaussian PDF at x=0
print(gaussian_pdf(0.0, mu=0, sigma=1))  # 0.3989...

# Binomial: P(3 heads in 5 fair flips)
print(binomial_pmf(3, 5, 0.5))  # 0.3125

# Poisson: P(k=2, λ=3)
print(poisson_pmf(2, 3.0))  # 0.2240...
