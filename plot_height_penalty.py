import numpy as np
import matplotlib.pyplot as plt
import math

def height_penalty(normalized_height):
    return (-2 / (1 + math.exp(-20 * (normalized_height - 0.4))) + 1) * 6

# Create x values (normalized heights from 0 to 1)
x = np.linspace(0, 1, 1000)

# Calculate y values (height penalties)
y = [height_penalty(h) for h in x]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Normalized Height')
plt.ylabel('Height Penalty')
plt.title('Height Penalty vs Normalized Height')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0.4, color='r', linestyle='--', alpha=0.5, label='Threshold (0.4)')
plt.legend()

# Show the plot
plt.show() 