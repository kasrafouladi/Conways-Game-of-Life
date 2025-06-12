import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

# Load NetCDF dataset
ds = xr.open_dataset("dataset/data_stream-mnth.nc")

# Select June, July, August and compute mean over the specified region
ds_summer = ds.sel(time=ds.time.dt.month.isin([6, 7, 8]))
global_mean = ds_summer['t2m'].mean(dim=['latitude', 'longitude'])

# Extract time (years) and temperature, clean NaNs
time = global_mean['time'].dt.year.values
temp = global_mean.values
temp_clean = temp[~np.isnan(temp)]
time_clean = time[~np.isnan(temp)]

# Select uniform interval points
start_year, end_year, step = input("please enter start_year, end_year and step for sample data. years should be in range [1950, 2020]:\n").split()
start_year = int(start_year)
end_year = int(end_year)
step = int(step)
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

# Local Newton Forward Interpolation (using 8 nearest neighbors)
def local_newton_forward(x, x_vals, y_vals, num_neighbors=8):
    idx = np.abs(x_vals - x).argmin()
    start = max(0, idx - num_neighbors//2)
    end = min(len(x_vals), start + num_neighbors)
    
    if end == len(x_vals):
        start = max(0, end - num_neighbors)
    
    sub_x = x_vals[start:end]
    sub_y = y_vals[start:end]
    
    if len(sub_x) < 2:
        return np.interp(x, x_vals, y_vals)
    
    h = sub_x[1] - sub_x[0]
    diff_table = forward_diff_table(sub_y)
    u = (x - sub_x[0]) / h
    res = sub_y[0]
    u_term = 1
    
    for i in range(1, len(sub_y)):
        u_term *= (u - (i - 1)) / i
        res += u_term * diff_table[i][0]
    
    return res

# Local Newton Backward Interpolation (using 8 nearest neighbors)
def local_newton_backward(x, x_vals, y_vals, num_neighbors=8):
    idx = np.abs(x_vals - x).argmin()
    start = max(0, idx - num_neighbors//2)
    end = min(len(x_vals), start + num_neighbors)
    
    if end == len(x_vals):
        start = max(0, end - num_neighbors)
    
    sub_x = x_vals[start:end]
    sub_y = y_vals[start:end]
    
    if len(sub_x) < 2:
        return np.interp(x, x_vals, y_vals)
    
    h = sub_x[1] - sub_x[0]
    diff_table = forward_diff_table(sub_y[::-1])
    u = (x - sub_x[-1]) / h
    res = sub_y[-1]
    u_term = 1
    
    for i in range(1, len(sub_y)):
        u_term *= (u + (i - 1)) / i
        res += u_term * diff_table[i][0]
    
    return res

# Local Lagrange interpolation using 8 nearest neighbors
def local_lagrange_interp(x, x_vals, y_vals, num_neighbors=8):
    idx = np.abs(x_vals - x).argmin()
    start = max(0, idx - num_neighbors//2)
    end = min(len(x_vals), start + num_neighbors)
    
    if end == len(x_vals):
        start = max(0, end - num_neighbors)
    
    sub_x = x_vals[start:end]
    sub_y = y_vals[start:end]
    
    if len(sub_x) < 2:
        return np.interp(x, x_vals, y_vals)
    
    poly = lagrange(sub_x, sub_y)
    return poly(x)

# Local Polynomial regression (using 8 nearest neighbors)
def local_poly_regression(x, x_vals, y_vals, degree=4, num_neighbors=8):
    idx = np.abs(x_vals - x).argmin()
    start = max(0, idx - num_neighbors//2)
    end = min(len(x_vals), start + num_neighbors)
    
    if end == len(x_vals):
        start = max(0, end - num_neighbors)
    
    sub_x = x_vals[start:end]
    sub_y = y_vals[start:end]
    
    if len(sub_x) < 2:
        return np.interp(x, x_vals, y_vals)
    
    actual_degree = min(degree, len(sub_x) - 1)
    
    x_mean, x_std = sub_x.mean(), sub_x.std()
    y_mean, y_std = sub_y.mean(), sub_y.std()
    
    if x_std == 0:
        return y_mean
    if y_std == 0:
        return sub_y[0]
    
    x_norm = (sub_x - x_mean) / x_std
    y_norm = (sub_y - y_mean) / y_std
    
    try:
        poly = Polynomial.fit(x_norm, y_norm, actual_degree)
        x_pred_norm = (x - x_mean) / x_std
        y_pred_norm = poly(x_pred_norm)
        return y_pred_norm * y_std + y_mean
    except:
        return np.interp(x, sub_x, sub_y)

# Prediction range
years_pred = np.arange(1950, 2021)

# Local interpolations
local_forward_pred = [local_newton_forward(x, uniform_years, uniform_temps) for x in years_pred]
local_backward_pred = [local_newton_backward(x, uniform_years, uniform_temps) for x in years_pred]
local_lagrange_pred = [local_lagrange_interp(x, uniform_years, uniform_temps) for x in years_pred]
local_reg_pred = [local_poly_regression(x, uniform_years, uniform_temps) for x in years_pred]

# Actual values
actual = np.interp(years_pred, time_clean, temp_clean)

# Plot: Each method compared with actual values individually
methods = {
    "Local Newton Forward": local_forward_pred,
    "Local Newton Backward": local_backward_pred,
    "Local Lagrange": local_lagrange_pred,
    "Local Polynomial Regression": local_reg_pred
}

for method_name, predicted_values in methods.items():
    plt.figure(figsize=(10, 5))
    plt.plot(years_pred, actual, label="Actual", linewidth=2)
    plt.plot(years_pred, predicted_values, label=method_name, linestyle='--')
    plt.xlabel("Year")
    plt.ylabel("Mean 2m Temperature (°C) [June, July, August]")
    plt.title(f"{method_name} vs Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
    plt.savefig(f"figs/{filename}_vs_actual[{start_year}, {end_year}, {step}].png")
    plt.show()

import pandas as pd
from sklearn.metrics import mean_squared_error

# Prepare data for CSV export
data = {
    "Year": years_pred,
    "Actual": actual,
    "Local_Newton_Forward": local_forward_pred,
    "Local_Newton_Backward": local_backward_pred,
    "Local_Lagrange": local_lagrange_pred,
    "Local_Poly_Regression": local_reg_pred
}

# Create DataFrame and export
results_df = pd.DataFrame(data)
results_df.to_csv(f"csv/local_t2m_predictions[{start_year}, {end_year}, {step}].csv", index=False)

# Calculate RMSE for each method
for method_name, predicted_values in methods.items():
    rmse = mean_squared_error(actual, predicted_values) ** 0.5
    print(f"RMSE for {method_name}: {rmse:.5f}")

print("done!")
