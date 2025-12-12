import matplotlib
print("Matplotlib version:", matplotlib.__version__)

# Force a non-GUI backend that only renders to files
# matplotlib.use("Agg")
print("Backend:", matplotlib.get_backend())

import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4]
y = [i**2 for i in x]

plt.figure()
plt.plot(x, y)
plt.title("Debug plot")
plt.xlabel("x")
plt.ylabel("x^2")

plt.savefig("debug_plot.png")
print("Saved debug_plot.png")