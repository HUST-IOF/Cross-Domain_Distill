import os
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import time

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ModifiedLeNet_v9(nn.Module):
    def __init__(self):
        super(ModifiedLeNet_v9, self).__init__()
        self.features_1 = nn.Sequential(
            DepthwiseSeparableConv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.features_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DepthwiseSeparableConv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DepthwiseSeparableConv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 32, 3)  # Output for classification
        )

        self.regressor = nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.features_1(x)
        regressor_output = self.regressor(x)
        x = self.features_2(x)
        x = x.view(x.size(0), -1)
        classification_output = self.classifier(x)
        return classification_output, regressor_output

# Assuming you have a list of Chinese labels
chinese_labels = ['锤子', '风炮', '钩机']

# Create label-to-index and index-to-label mappings
label_to_index = {label: idx for idx, label in enumerate(chinese_labels)}
index_to_label = {idx: label for idx, label in enumerate(chinese_labels)}


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.categories = sorted(os.listdir(root_dir))
        self.data = self.load_data()

    def load_data(self):
        data = []
        for category in self.categories:
            category_path = os.path.join(self.root_dir, category)
            file_list = [os.path.join(category_path, file) for file in os.listdir(category_path) if
                         file.endswith('.npy')]
            data.extend([(file, category) for file in file_list])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        matrix = np.load(file_path)

        if self.transform:
            matrix = self.transform(matrix)

        # Encode labels to numerical indices
        label = label_to_index[label]

        return {'matrix': matrix, 'label': label}


def check_accuracy(loader, criterion, model, device):
    model.eval()
    correct_num = 0
    total_num = 0
    total_loss = []

    with torch.no_grad():
        for batch in loader:
            matrices, targets = batch['matrix'].to(device), batch['label'].to(device)
            matrices = matrices.to(torch.float32)  # or torch.float64 based on your data type
            outputs, regressor_feature_map = model(matrices)

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            total_num += targets.size(0)
            correct_num += (predicted == targets).sum().item()

    checked_accuracy = correct_num / total_num
    checked_loss = sum(total_loss) / len(total_loss)
    # print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return checked_accuracy, checked_loss


def main():
    # Create an instance of the model
    # model = LeNet_relu()

    # List of paths to the .pt files
    model_paths = ['../trained_models/slimnn_student_fold_1.pt',
                   '../trained_models/slimnn_student_fold_2.pt',
                   '../trained_models/slimnn_student_fold_3.pt',
                   '../trained_models/slimnn_student_fold_4.pt',
                   '../trained_models/slimnn_student_fold_5.pt']

    # Placeholder for original model's test accuracies
    original_test_acc_list = []
    original_train_acc_list = []

    unquan_train_inference_times = []
    unquan_test_inference_times = []

    # Define a transformation to convert matrices to PyTorch tensors
    # Create instances of the dataset for training, validation, and test sets
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(root_dir='../example_samples/train', transform=data_transform)

    test_dataset = CustomDataset(root_dir='../example_samples/test', transform=data_transform)

    batch_size = 8

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    for index, path in enumerate(model_paths, start=1):
        print(f"Fold {index}:")

        # Create an instance of the model
        model = ModifiedLeNet_v9()

        # Load the state dictionary from the file and apply it to the model
        model.load_state_dict(torch.load(path))

        model.to(device)

        # Test the final model on the train and test set

        # Measure the time for model inference
        start_time = time.time()

        original_train_accuracy, original_train_loss = check_accuracy(train_loader, criterion, model, device)

        inference_time = time.time() - start_time

        unquan_train_inference_times.append(inference_time)

        original_train_acc_list.append(original_train_accuracy)

        # Print the inference time
        print("Unquantized Model Train Inference Time: {:.4f} seconds".format(inference_time))

        print(f"Original Train Accuracy: {original_train_accuracy:.4f}, Test Loss: {original_train_loss:.4f}")

        # Measure the time for model inference
        start_time = time.time()

        original_test_accuracy, original_test_loss = check_accuracy(test_loader, criterion, model, device)

        inference_time = time.time() - start_time

        unquan_test_inference_times.append(inference_time)

        original_test_acc_list.append(original_test_accuracy)

        # Print the inference time
        print("Unquantized Model Train Inference Time: {:.4f} seconds".format(inference_time))

        print(f"Original Test Accuracy: {original_test_accuracy:.4f}, Test Loss: {original_test_loss:.4f}")


    # Calculate the average inference time
    unquan_avg_train_inference_time = np.mean(unquan_train_inference_times) / len(train_loader)
    print("Unquantized Model Average Test Inference Time: {:.6f} seconds".format(unquan_avg_train_inference_time))

    unquan_avg_test_inference_time = np.mean(unquan_test_inference_times) / len(test_loader)
    print("Unquantized Model Average Test Inference Time: {:.6f} seconds".format(unquan_avg_test_inference_time))

    # Calculate the average of the original train accuracies
    avg_original_train_accuracy = np.mean(original_train_acc_list)

    print("Average Original Training Accuracy:", avg_original_train_accuracy)

    # Calculate the average of the original test accuracies
    avg_original_test_accuracy = np.mean(original_test_acc_list)

    print("Average Original Testing Accuracy:", avg_original_test_accuracy)


if __name__ == "__main__":
    main()