"""Statistics and regression demo."""

from cds.stats import (
    correlation,
    linear_regression,
    mean,
    median,
    stdev,
    variance,
)

# --- Descriptive statistics ---


def main() -> None:
    data = [23.1, 27.5, 19.8, 31.2, 25.6, 28.3, 22.0, 30.1]
    print("=== Descriptive Statistics ===")
    print(f"Data: {data}")
    print(f"Mean:     {mean(data):.2f}")
    print(f"Median:   {median(data):.2f}")
    print(f"Variance: {variance(data):.2f}")
    print(f"Std Dev:  {stdev(data):.2f}")

    # --- Correlation ---
    print("\n=== Correlation ===")
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y = [2.1, 4.3, 5.8, 8.1, 9.7, 12.2, 13.8, 16.1]
    r = correlation(x, y)
    print(f"r = {r:.4f}  (1.0 = perfect positive)")

    # --- Linear regression ---
    print("\n=== Linear Regression ===")
    reg = linear_regression(x, y)
    print(f"y = {reg.slope:.2f}x + {reg.intercept:.2f}")
    print(f"R² = {reg.r_squared:.4f}")
    print(f"Prediction at x=10: {reg.predict(10):.2f}")


if __name__ == "__main__":
    main()
