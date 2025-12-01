import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Step': [1, 2, 3, 4, 5, 6],
    'Training Loss': [0.0, 0.1046, 0.0, 0.0, 0.0010, 0.0109]
}

df = pd.DataFrame(data)

# If you saved the notepad file as a CSV, e.g., "loss.csv"
# df = pd.read_csv("loss.csv", sep="\t")  # use sep="\t" if tab-separated

# Plot
plt.plot(df['Step'], df['Training Loss'], marker='o')
plt.xlabel('Step')
plt.ylabel('Training Loss')
plt.title('Training Loss over Steps')
plt.grid(True)
plt.show()
