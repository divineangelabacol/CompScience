import matplotlib.pyplot as plt

# Sample data
x = [10, 20, 30, 40, 50, 60, 70]
y = [20, 40, 50, 70, 90, 95, 100]

# Calculate the means of x and y
mean_x = sum(x) / len(x)
mean_y = sum(y) / len(y)

# Calculate the slope (m) and intercept (b)
n = len(x)
numerator = sum([x[i] * y[i] for i in range(n)]) - n * mean_x * mean_y
denominator = sum([x[i] ** 2 for i in range(n)]) - n * mean_x ** 2
m = numerator / denominator
b = mean_y - m * mean_x

# Print the calculated slope and intercept
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

# Predict y values
y_pred = [m * xi + b for xi in x]

# Plot the data points
plt.scatter(x, y, color="green", label="Data Points")

# Plot the regression line
plt.plot(x, y_pred, color="blue", label="Regression Line")

# Add labels, title, and legend
plt.xlabel("X")
plt.ylabel("Y")
plt.title( "Linear Regression")
plt.legend()

# Show the plot
plt.show()
