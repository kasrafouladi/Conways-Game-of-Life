import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

# Load NetCDF dataset
ds = xr.open_dataset("gistemp1200_GHCNv4_ERSSTv5.nc")

# Compute global mean temperature anomaly
global_mean = ds['tempanomaly'].mean(dim=['lat', 'lon'])
time = global_mean['time'].dt.year.values
temp = global_mean.values

# Remove NaNs
temp_clean = temp[~np.isnan(temp)]
time_clean = time[~np.isnan(temp)]

# Select uniform interval points for Newton interpolation
start_year = 1950
end_year = 2020
step = 5
uniform_years = np.arange(start_year, end_year + 1, step)

uniform_temps = np.interp(uniform_years, time_clean, temp_clean)

# Forward difference table
def forward_diff_table(y):
    n = len(y)
    diff_table = [y.copy()]
    for i in range(1, n):
        diff = [diff_table[i-1][j+1] - diff_table[i-1][j] for j in range(n - i)]
        diff_table.append(diff)
    return diff_table

# Newton Forward Interpolation
def newton_forward(x, x_vals, y_vals):
    h = x_vals[1] - x_vals[0]
    diff_table = forward_diff_table(y_vals)
    u = (x - x_vals[0]) / h
    res = y_vals[0]
    u_term = 1
    for i in range(1, len(y_vals)):
        u_term *= (u - (i - 1)) / i
        res += u_term * diff_table[i][0]
    return res

# Newton Backward Interpolation
def newton_backward(x, x_vals, y_vals):
    h = x_vals[1] - x_vals[0]
    diff_table = forward_diff_table(y_vals[::-1])
    u = (x - x_vals[-1]) / h
    res = y_vals[-1]
    u_term = 1
    for i in range(1, len(y_vals)):
        u_term *= (u + (i - 1)) / i
        res += u_term * diff_table[i][0]
    return res

# Apply Lagrange interpolation
def lagrange_interp(x, x_vals, y_vals):
    poly = lagrange(x_vals, y_vals)
    return poly(x)

# Polynomial regression
def poly_regression(x_vals, y_vals, degree):
    coeffs = Polynomial.fit(x_vals, y_vals, degree).convert().coef
    return coeffs

def eval_poly(x, coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

# Generate prediction years
years_pred = np.arange(1950, 2030)

# Interpolations
forward_pred = [newton_forward(x, uniform_years, uniform_temps) for x in years_pred]
backward_pred = [newton_backward(x, uniform_years, uniform_temps) for x in years_pred]
lagrange_pred = [lagrange_interp(x, uniform_years, uniform_temps) for x in years_pred]

# Regression
reg_coeffs = poly_regression(time_clean, temp_clean, 10)
reg_pred = [eval_poly(x, reg_coeffs) for x in years_pred]

# Actual values for comparison
actual = np.interp(years_pred, time_clean, temp_clean)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(years_pred, actual, label="Actual", linewidth=2)
plt.plot(years_pred, forward_pred, label="Newton Forward", linestyle='--')
plt.plot(years_pred, backward_pred, label="Newton Backward", linestyle='--')
plt.plot(years_pred, lagrange_pred, label="Lagrange", linestyle='--')
plt.plot(years_pred, reg_pred, label="Polynomial Regression (deg=10)", linestyle='--')
plt.xlabel("Year")
plt.ylabel("Global Mean Temperature Anomaly")
plt.title("Comparison of Interpolation and Regression Methods")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
