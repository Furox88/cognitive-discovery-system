from cds.stats import mean, stdev, correlation, linear_regression

data = [12.5, 14.3, 11.8, 15.1, 13.7]
print(f"mean={mean(data):.2f}, std={stdev(data):.2f}")

x = [1, 2, 3, 4, 5]
y = [2.1, 3.9, 6.2, 7.8, 10.1]
reg = linear_regression(x, y)
print(f"y = {reg.slope:.2f}x + {reg.intercept:.2f}, R²={reg.r_squared:.3f}")
