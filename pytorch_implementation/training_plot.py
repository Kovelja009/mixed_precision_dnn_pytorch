import pandas as pd
import matplotlib.pyplot as plt
import sys


# Type of command:
#  python training_plot.py experiments/PARAMETRIC_FP_WAQ_DELTA_XMAX_INIT_ADAM
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python training_plot.py experiments/<experiment_name>")
        exit(1)

    experiment_name = sys.argv[1]

    # Read in the data
    train_data = pd.read_csv(f'{experiment_name}/train_losses.csv')
    test_data = pd.read_csv(f'{experiment_name}/test_losses.csv')

    plt.figure(figsize=(10, 6))
    plt.plot(train_data['epoch'], train_data['loss'], label='Train Loss', marker='o', color='blue')
    plt.plot(test_data['epoch'], test_data['loss'], label='Test Loss', marker='x', color='red')
    plt.title('Train Loss and Test Loss over iterations (uniform quantization with learnable delta and xmax)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
