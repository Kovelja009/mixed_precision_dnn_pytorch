import pandas as pd
import matplotlib.pyplot as plt
import sys


def plot_one_model(experiment_name):
    # Read in the data
    train_data = pd.read_csv(f'{experiment_name}/train_losses_q.csv')
    test_data = pd.read_csv(f'{experiment_name}/test_losses_q.csv')

    plt.figure(figsize=(10, 6))
    plt.plot(train_data['epoch'], train_data['loss'], label='Train Loss', marker='o', color='blue')
    plt.plot(test_data['epoch'], test_data['loss'], label='Test Loss', marker='x', color='red')
    plt.title('Train Loss and Test Loss over iterations (uniform quantization with learnable delta and b)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_compare_models(experiment_name):
    # Read loss data
    train_data = pd.read_csv(f'{experiment_name}/train_losses.csv')
    test_data = pd.read_csv(f'{experiment_name}/test_losses.csv')
    train_data_q = pd.read_csv(f'{experiment_name}/train_losses_q.csv')
    test_data_q = pd.read_csv(f'{experiment_name}/test_losses_q.csv')

    # Read accuracy data
    train_acc_data = pd.read_csv(f'{experiment_name}/train_accs.csv')
    test_acc_data = pd.read_csv(f'{experiment_name}/test_accs.csv')
    train_acc_data_q = pd.read_csv(f'{experiment_name}/train_accs_q.csv')
    test_acc_data_q = pd.read_csv(f'{experiment_name}/test_accs_q.csv')

    # Plot losses
    plt.figure(figsize=(15, 6))

    # Plotting Losses
    plt.subplot(1, 2, 1)
    plt.plot(train_data['epoch'], train_data['loss'], label='Train Loss', marker='o', color='blue')
    plt.plot(test_data['epoch'], test_data['loss'], label='Test Loss', marker='x', color='red')
    plt.plot(train_data_q['epoch'], train_data_q['loss'], label='Train Loss Quantized', marker='o', color='green')
    plt.plot(test_data_q['epoch'], test_data_q['loss'], label='Test Loss Quantized', marker='x', color='orange')
    plt.title('Train Loss and Test Loss over iterations')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plotting Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_data['epoch'], train_acc_data['accuracy'], label='Train Accuracy', marker='o', color='blue')
    plt.plot(test_acc_data['epoch'], test_acc_data['accuracy'], label='Test Accuracy', marker='x', color='red')
    plt.plot(train_acc_data_q['epoch'], train_acc_data_q['accuracy'], label='Train Accuracy Quantized', marker='o',
             color='green')
    plt.plot(test_acc_data_q['epoch'], test_acc_data_q['accuracy'], label='Test Accuracy Quantized', marker='x',
             color='orange')
    plt.title('Train Accuracy and Test Accuracy over iterations')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Type of command:
#  python training_plot.py experiments/PARAMETRIC_FP_WAQ_DELTA_XMAX_RLR
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python training_plot.py experiments/<experiment_name>")
        exit(1)

    experiment_name = sys.argv[1]
    plot_one_model(experiment_name)

