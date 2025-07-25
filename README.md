
# Federated Learning Simulation for Medical Diagnosis

This project provides a simulation of a Federated Learning (FL) system for a medical diagnosis task using synthetic patient data. It demonstrates how multiple local models can be trained on decentralized data and aggregated into a single, more robust global model without sharing the raw, sensitive patient information.

The entire simulation is contained within a single Python script and uses PyTorch for building and training the neural network models.

---
## Table of Contents
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Code Breakdown](#code-breakdown)
- [Expected Output](#expected-output)
---

## Key Features

- **Synthetic Data Generation**: Creates a synthetic tabular dataset of patient records with features like age, gender, and medical history, including a class imbalance to mimic real-world scenarios.
- **Federated Learning Simulation**: Implements the core Federated Learning loop with multiple local clients.
- **Federated Averaging (FedAvg)**: Uses the FedAvg algorithm to aggregate model weights from local clients to update the global model.
- **PyTorch Neural Network**: A simple neural network for binary classification built with PyTorch.
- **Performance Evaluation**: Measures the global model's performance using accuracy and loss metrics.
- **Visualization**: Generates plots for:
    - Global model accuracy and loss over federated rounds.
    - A confusion matrix for the final global model to assess its classification performance.
---

## How It Works

The simulation follows these steps:
1.  **Data Creation**: A synthetic CSV file (`cleaned_synthetic_patient_data.csv`) is generated with imbalanced patient data.
2.  **Data Loading**: The data is loaded, preprocessed (e.g., feature scaling), and split into training and validation sets.
3.  **Initialization**: A global model and several "local" models (in this case, 3) are initialized with the same architecture.
4.  **Federated Rounds**: The simulation runs for a set number of rounds. In each round:
    a. Each local model is trained on the same batch of data (simulating training on its own private data).
    b. The weights from all trained local models are sent to a central server.
    c. The server performs **Federated Averaging** to create an updated set of weights for the global model.
5.  **Evaluation**: After all rounds are complete, the final global model is evaluated. Its accuracy is calculated, and a confusion matrix is displayed to show its performance in distinguishing between the two diagnostic classes.
6.  **Reporting**: A final summary report and performance graphs are generated.
---

## Prerequisites

This script requires Python 3.x and the following libraries:
- pandas
- numpy
- torch (PyTorch)
- scikit-learn
- matplotlib
---

## Installation

1.  **Clone the repository (or save the script):**
    If this is a git repository, clone it:
    ```bash
    git clone  https://github.com/ARGEN-FIRE/Federated-Learning-in-Healthcare.git
    cd Federated-Learning-in-Healthcare
    ```
    Alternatively, save the script provided as `federated_simulation.py`.

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    You can create a `requirements.txt` file with the following content:
    ```txt
    pandas
    numpy
    torch
    scikit-learn
    matplotlib
    ```
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
---

## How to Run

Execute the Python script from your terminal:
```bash
python federated_simulation.py
```

The script will run the full simulation, print the progress for each round, and finally display the performance plots and summary report.

---
# Project Structure

When you run the script, your directory will contain:

```bash
.
├── federated_simulation.py               # The main Python script
├── cleaned_synthetic_patient_data.csv    # The synthetic data file generated by the script
└── README.md                             # This README file
```

- **create_synthetic_patient_data():** Generates a Pandas DataFrame with synthetic data. It intentionally creates class imbalance and ensures data consistency (e.g., a patient with no medical history flags cannot have a positive diagnosis).
- **Data Preprocessing:** The main part of the script loads the data, uses StandardScaler to normalize the 'Age' feature, and splits the data using train_test_split.
- **MedicalDataset & DataLoader:** Standard PyTorch classes to handle dataset loading and batching for the model.
- **MultiModalMedicalNN:** A simple feed-forward neural network.

**Note:** Although named "MultiModal", this implementation only uses a single modality (tabular data). The architecture is flexible enough to be extended for multi-modal inputs.

---
- **train_local_model():** Simulates the training process on a single client. It runs for a set number of epochs and returns the training losses and accuracies.
- **federated_averaging():** The core aggregation algorithm. It averages the weights (state_dict) of all local models to update the global model.
- **federated_learning():** The main control loop that orchestrates the rounds, calls local training, and triggers the federated averaging.
- **evaluate_and_display_confusion_matrix():** Evaluates the final global model on the training data and plots a confusion matrix to visualize its predictive performance.
- **generate_report() & plot_graphs():** Helper functions for generating a final text summary and plotting the training progress.
- **Note:** The accuracy printed during local training (accuracy_increase = ...) is synthetically controlled to show a smooth progression for demonstration purposes. The actual model training and loss calculation are genuine.
---

# Expected Output

```bash
When you run the script, you will see:
Console Output: Logs showing the progress of each federated round, including the average loss and accuracy, followed by a final evaluation and summary.
```
```bash
Round 1
Epoch 1/30, Loss: 21.9023, Accuracy: 59.13%
...
Epoch 30/30, Loss: 19.8242, Accuracy: 92.00%
... (for each local model) ...
Round 1 - Avg Loss: 20.8066, Avg Accuracy: 75.50%
...
Round 5 - Avg Loss: 20.2589, Avg Accuracy: 75.50%

Evaluating the global model...
Model Accuracy: 89.81%
```
---

**Federated Learning Summary Report**
- Global Accuracies: [0.755, 0.755, 0.755, 0.755, 0.755]
- Global Losses: [20.8066..., 20.4283..., ..., 20.2589...]

---
- **Generated File:** A cleaned_synthetic_patient_data.csv file will be created in the same directory.
- **Plots:** Two plot windows will appear at the end of the script:
- **Confusion Matrix:** A matrix showing the True Positives, True Negatives, False Positives, and False Negatives of the final global model.
- **Global Performance Over Rounds:** A side-by-side plot showing how the global model's loss decreased and its accuracy evolved across the federated rounds.
