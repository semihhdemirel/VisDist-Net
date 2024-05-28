import torch
from torchvision import datasets, transforms
from models.studentModel import CombinedModel
import os
import engine
from models.vit import ViT

# Set device and hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_of_fold = 5

for i in range(1, num_of_fold+1):
    DATA_PATH = './fold_dataset'
    print("DATA_PATH = ", DATA_PATH)

    TRAIN_PATH = os.path.join(DATA_PATH, "Fold" + str(i), "Train/")
    print("TRAIN_PATH = ", TRAIN_PATH)

    VAL_PATH = os.path.join(DATA_PATH, "Fold" + str(i),  "Val/")
    print("TEST_PATH = ",VAL_PATH)

    TEST_PATH = os.path.join(DATA_PATH, "Fold" + str(i),  "Test/")
    print("TEST_PATH = ",TEST_PATH)

    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]
    image_size = 400

    data_transforms = {"train":transforms.Compose([
                                    transforms.Resize((image_size, image_size)), #Resizes all images into same dimension
                                    transforms.ToTensor(), # Coverts into Tensors
                                    transforms.Normalize(mean = mean_nums, std=std_nums)]), # Normalizes
                        "val": transforms.Compose([
                                    transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean_nums, std = std_nums)
                        ]),
                        "test": transforms.Compose([
                                    transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean_nums, std = std_nums)
                        ])                    
                        }

    train_data = datasets.ImageFolder(TRAIN_PATH,       
                        transform=data_transforms['train'])

    val_data = datasets.ImageFolder(VAL_PATH,
                                    transform = data_transforms['val'])

    test_data = datasets.ImageFolder(TEST_PATH,
                                    transform = data_transforms['test'])

    lr_teacher = 0.001
    lr_student = 0.001
    temperature = 2.0
    num_epochs = 50
    batch = 4

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch,
                                            shuffle = True, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch,
                                        shuffle = False, num_workers = 0)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch,
                                        shuffle = False, num_workers = 0)

    teacher_checkpoint = torch.load(f"./fold_outputs/Fold{str(i)}_vit_best.pth")
    teacher_model = ViT(
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
    teacher_model.load_state_dict(teacher_checkpoint)
    teacher_model = teacher_model.to(device)
    # Instantiate the lightweight network:
    torch.manual_seed(42)
    student_model = CombinedModel(num_classes=6).to(device)

    engine.train_knowledge_distillation(teacher=teacher_model, student=student_model, train_loader=train_loader, val_loader=val_loader, epochs=num_epochs, learning_rate=lr_teacher, T=temperature, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
