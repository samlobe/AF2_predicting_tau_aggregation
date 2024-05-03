import matplotlib.pyplot as plt

# Replace 'your_file_path.dat' with the path to your data file
file_path = 'request.fasta.seq.aggr_profile.dat'
with open(file_path, 'r') as file:
    content = file.readlines()

# Convert the string data to float
data_points = [float(line.strip()) for line in content]

# Plotting the data
plt.figure(figsize=(10, 5))
plt.plot(data_points, marker='o', linestyle='-', color='b')
plt.xlim(left=200,right=430)
plt.title('PASTA aggregation probability')
plt.xlabel('tau residue id')
plt.ylabel('Probability')
plt.grid(True)
plt.show()

