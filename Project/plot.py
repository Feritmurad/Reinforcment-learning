import matplotlib.pyplot as plt
import os

def plot_returns_from_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            episodes = []
            scorelist = []

            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts[0] == 'RETURN':
                        episodes.append(int(parts[1]))
                        scorelist.append(float(parts[2]))

            # Plot the returns using matplotlib
            plt.figure()
            plt.plot(episodes, scorelist, label='Score')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.title(f'Training Returns Over Episodes for {file_name}')
            plt.legend()
            plt.grid(True)
            # Save the plot as a PNG file, keeping the original file name
            plot_file_name = f'{file_name}.png'
            plt.savefig(os.path.join(folder_path, plot_file_name))
            plt.show()  # Display the plot

# Plot the returns from all log files in the folder
plot_returns_from_folder('population methods')
