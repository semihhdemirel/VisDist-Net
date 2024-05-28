import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def save_best_model(model, i):
    model_name = type(model).__name__.lower()
    model_filename = f'./fold_outputs/Fold{i}_{model_name}_best.pth'
    torch.save(model.state_dict(), model_filename)

def train(model, train_loader, val_loader, i, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    best_loss = np.inf
    val_loss_list = []
    train_loss_list = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        # Training
        running_loss = 0.0
        tqdm_loader = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} (Training)', unit='batch')

        for inputs, labels in tqdm_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss / len(train_loader)}")
        train_loss_list.append(running_loss / len(train_loader))
        print("Train Loss List:",train_loss_list)
        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            tqdm_loader = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} (Validation)', unit='batch')

            for val_inputs, val_labels in tqdm_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()

            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}")
            val_loss_list.append(val_loss)
            print("Val Loss List", val_loss_list)
        # Save the model if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            save_best_model(model, i)

    model.train()  # Set the model back to training mode after validation

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        tqdm_loader = tqdm(test_loader, desc='Testing', unit='batch')

        for inputs, labels in tqdm_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            tqdm_loader.set_postfix(accuracy=100 * correct / total)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def train_knowledge_distillation(i, teacher, student, train_loader, val_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device, save_path="./outputs_kd"):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
 
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    best_loss = float('inf')
    val_loss_list = []
    train_loss_list = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        tqdm_loader = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')

        for inputs, labels in tqdm_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            # Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tqdm_loader.set_postfix(loss=running_loss / len(tqdm_loader))

        # Validation
        val_loss = validate(student, val_loader, ce_loss, device)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}")
        train_loss_list.append(running_loss / len(train_loader))
        val_loss_list.append(val_loss)
        print("Train Loss List:",train_loss_list)
        print("Val Loss List", val_loss_list)

        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = f'./fold_outputs/Fold{i}_KD_{type(student).__name__.lower()}_best.pth'
            torch.save(student.state_dict(), save_path)
            print("Best model saved.")

    print("Training finished.")

def validate(model, val_loader, ce_loss, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)

            # Calculate the true label loss
            label_loss = ce_loss(logits, labels)

            val_loss += label_loss.item()

    model.train()
    return val_loss / len(val_loader)
