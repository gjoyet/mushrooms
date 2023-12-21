import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import seaborn as sns


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Define the layers
        self.layer1 = nn.Linear(22, 64)  # 22 input features, 64 hidden units
        self.layer2 = nn.Linear(64, 128)  # 64 hidden units, 128 hidden units
        self.layer3 = nn.Linear(128, 64)  # 128 hidden units, 64 hidden units
        self.layer4 = nn.Linear(64, 32)  # 64 hidden units, 32 hidden units
        self.layer5 = nn.Linear(32, 1)  # 32 hidden units, 1 output units

    def forward(self, x):
        # Define the forward pass
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.sigmoid(self.layer5(x))
        return x


def load_data(path):
    df = pd.read_csv(path)

    # Assuming the last two columns are the labels
    # Adjust if your data has a different structure
    predictors = df.columns[1:]

    X = df[predictors].copy()
    y = df['class'].copy()

    for c in X.columns:
        X[c] = X[c].astype('category').cat.codes

    y = y.astype('category').cat.codes

    # Convert to PyTorch tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.long)

    # Create a TensorDataset
    dataset = TensorDataset(X, y)

    return dataset


def train_network(dataloader, rel_weight):
    # Instantiate the neural network
    model = NeuralNetwork()

    # Define the loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 15

    for epoch in range(num_epochs):
        train_loss = 0

        for inputs, labels in dataloader:
            y_train = labels[:, None]
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            rel_weights = torch.tensor([1 if lab == 0 else rel_weight for lab in labels])
            criterion = nn.BCELoss(weight=torch.reshape(rel_weights, (len(rel_weights), 1)))
            loss = criterion(outputs, y_train.float())
            train_loss += loss.item() * inputs.size(0)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

    return model


if __name__ == '__main__':

    # Assuming your CSV file has columns for features and labels
    # Adjust column names accordingly
    train_data = load_data("../data/mushrooms_train.csv")
    test_data = load_data("../data/mushrooms_test.csv")

    train_data_size = train_data.tensors[0].shape[0]
    test_data_size = test_data.tensors[0].shape[0]

    X_test = test_data.tensors[0]
    y_test = test_data.tensors[1]

    # Create a DataLoader
    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    with sns.axes_style("white"):
        fig, axs = plt.subplots(3, 2, figsize=(14, 18))
        axs = axs.flatten()
        fig.suptitle('Confusion matrices for different relative weighting w of the poisonous class')
        plt.rcParams.update({'font.size': 12})

        fp = []
        fn = []
        acc = []

        weights = [1, 2, 5, 10, 25, 50]

        for i, w in enumerate(weights):
            model = train_network(train_dataloader, w)
            test_outputs = model(X_test)

            y_pred = (test_outputs > 0.5).int()
            conf_mat = confusion_matrix(y_test, y_pred, normalize="true")
            cmd = ConfusionMatrixDisplay(conf_mat, display_labels=["edible", "poisonous"])
            cmd.plot(cmap=plt.cm.Greens, values_format='.6f', ax=axs[i])
            for labels in cmd.text_.ravel():
                labels.set_fontsize(16)
            axs[i].title.set_text('w = {}'.format(w))

            fp.append(conf_mat[0, 1])
            fn.append(conf_mat[1, 0])
            acc.append(accuracy_score(y_test, y_pred))

        print(fp)
        print(fn)
        print(acc)

        plt.show()

    # Save the trained model if needed
    torch.save(model.state_dict(), 'trained_model.pth')
