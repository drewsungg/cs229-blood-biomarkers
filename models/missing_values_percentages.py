import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./bloodDataNAN.csv', sep='\t')

# Check for missing values in each column
missing_values = df.isna().sum()
total_values = df.size
percentage_missing = (missing_values / total_values)

print(percentage_missing)

# Plot
plt.figure(figsize=(10, 8))
percentage_missing.plot(kind='bar')
plt.ylabel('Percentage of Missing Values')
plt.xlabel('Features')
plt.title('Percentage of Missing Values by Feature')

# Save the figure
plt.savefig('missing_values.png', dpi=300)

# Display the plot
plt.show()