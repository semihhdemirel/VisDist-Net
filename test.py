import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import time
from models.MobileNet import MobileNetV1 
from models.ResNet18 import ResNet18 
from models.studentModel import CombinedModel 
from models.vit import ViT


from sklearn.metrics import precision_score, recall_score, f1_score

def test(model, dataloader, device, model_filename):
    checkpoint = torch.load(model_filename)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    total_samples = 0
    total_correct = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        tqdm_loader = tqdm(dataloader, desc='Testing', unit='batch')
        for inputs, labels in tqdm_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = total_correct / total_samples

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1

import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_of_folds = 5
for i in range(1, num_of_folds+1):
    print("--------------------------------------------------------------------------------------")

    DATA_PATH = './fold_dataset'
    print("DATA_PATH = ",DATA_PATH)

    TRAIN_PATH = os.path.join(DATA_PATH, "Fold" + str(i), "Train/")
    print("TRAIN_PATH = ", TRAIN_PATH)

    VAL_PATH = os.path.join(DATA_PATH, "Fold" + str(i),  "Val/")
    print("TEST_PATH = ",VAL_PATH)

    TEST_PATH = os.path.join(DATA_PATH, "Fold" + str(i),  "Test/")
    print("TEST_PATH = ",TEST_PATH)

    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]

    image_size = 400


    data_transforms = {"test": transforms.Compose([
                                    transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean_nums, std = std_nums)
                        ])                    
                        }

    test_data = datasets.ImageFolder(TEST_PATH,
                                    transform = data_transforms['test'])
    batch = 8
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch,
                                        shuffle = False, num_workers = 0)
    best_model = ViT(
        image_size = image_size,
        patch_size = 16,
        num_classes = 6,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(device)
    
    vit_start = time.time()
    vit_accuracy, vit_precision, vit_recall, vit_f1 = test(best_model, test_loader, device, model_filename=f'./ckpts/Fold{str(i)}_vit_best.pth')
    vit_end = time.time()
    vit_elapsed = vit_end-vit_start
    print(f"RESULT OF: Fold{str(i)}_vit_best.pth")
    print(f"Accuracy: {vit_accuracy:.4f}%")
    print(f"Precision: {vit_precision:.4f}")
    print(f"Recall: {vit_recall:.4f}")
    print(f"F1 Score: {vit_f1:.4f}")
    print(f"Elapsed time: {vit_elapsed:.4f} seconds")
        
    best_model = MobileNetV1(num_classes=6)
    mobilenetv1_start = time.time()
    mobilenetv1_accuracy, mobilenetv1_precision, mobilenetv1_recall, mobilenetv1_f1 = test(best_model, test_loader, device, model_filename=f'./ckpts/Fold{str(i)}_mobilenetv1_best.pth')
    mobilenetv1_end = time.time()
    mobilenetv1_elapsed = mobilenetv1_end-mobilenetv1_start
    print(f"\nRESULT OF: Fold{str(i)}_mobilenetv1_best.pth")
    print(f"Accuracy: {mobilenetv1_accuracy:.4f}%")
    print(f"Precision: {mobilenetv1_precision:.4f}")
    print(f"Recall: {mobilenetv1_recall:.4f}")
    print(f"F1 Score: {mobilenetv1_f1:.4f}")
    print(f"Elapsed time: {mobilenetv1_elapsed:.4f} seconds")


    best_model = ResNet18(num_classes=6)
    resnet18_start = time.time()
    resnet18_accuracy, resnet18_precision, resnet18_recall, resnet18_f1 = test(best_model, test_loader, device, model_filename=f'./ckpts/Fold{str(i)}_resnet18_best.pth')
    resnet18_end = time.time()
    resnet18_elapsed = resnet18_end-resnet18_start
    print(f"\nRESULT OF: Fold{str(i)}_resnet18_best.pth")
    print(f"Accuracy: {resnet18_accuracy:.4f}%")
    print(f"Precision: {resnet18_precision:.4f}")
    print(f"Recall: {resnet18_recall:.4f}")
    print(f"F1 Score: {resnet18_f1:.4f}")
    print(f"Elapsed time: {resnet18_elapsed:.4f} seconds")
    

    best_model = CombinedModel(num_classes=6)
    combinedmodel_start = time.time()
    combinedmodel_accuracy, combinedmodel_precision, combinedmodel_recall, combinedmodel_f1 = test(best_model, test_loader, device, model_filename=f'./ckpts/Fold{str(i)}_combinedmodel_best.pth')
    combinedmodel_end = time.time() 
    combinedmodel_elapsed = combinedmodel_end-combinedmodel_start
    print(f"\nRESULT OF: Fold{str(i)}_combinedmodel_best.pth")
    print(f"Accuracy: {combinedmodel_accuracy:.4f}%")
    print(f"Precision: {combinedmodel_precision:.4f}")
    print(f"Recall: {combinedmodel_recall:.4f}")
    print(f"F1 Score: {combinedmodel_f1:.4f}")
    print(f"Elapsed time: {combinedmodel_elapsed:.4f} seconds")

    
    best_model = CombinedModel(num_classes=6)
    KD_combinedmodel_start = time.time()
    KD_combinedmodel_accuracy, KD_combinedmodel_precision, KD_combinedmodel_recall, KD_combinedmodel_f1 = test(best_model, test_loader, device, model_filename=f'./ckpts/Fold{str(i)}_KD_combinedmodel_best.pth')
    KD_combinedmodel_end = time.time()
    KD_combinedmodel_elapsed = KD_combinedmodel_end-KD_combinedmodel_start
    print(f"\nRESULT OF: Fold{str(i)}_KD_combinedmodel_best.pth")
    print(f"Accuracy: {KD_combinedmodel_accuracy:.4f}%")
    print(f"Precision: {KD_combinedmodel_precision:.4f}")
    print(f"Recall: {KD_combinedmodel_recall:.4f}")
    print(f"F1 Score: {KD_combinedmodel_f1:.4f}")
    print(f"Elapsed time: {KD_combinedmodel_elapsed:.4f} seconds")

    
    print("--------------------------------------------------------------------------------------")
