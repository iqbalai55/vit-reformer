import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

# Import custom utilities (assume these are defined in utils folder)
from utils.datasets import get_lpfw_dataloaders, get_student_dataloader
from utils.loss import FocalLoss


def initialize_model(model_type, num_classes, device):
    """Initialize the model based on the selected type."""
    if model_type == "virface":
        from reformer.reformer_pytorch import ViRWithArcMargin
        model = ViRWithArcMargin(
            image_size=224, 
            patch_size=8, 
            num_classes=num_classes, 
            dim=256, 
            depth=12, 
            heads=8, 
            arc_s=64.0, 
            arc_m=0.50,
            n_hashes=1,
            bucket_size=5
        )
    elif model_type == "vir":
        from reformer.vir_pytorch import ViR
        model = ViR(
            img_size=224,
            patch_size=8,
            in_channels=3,  
            num_classes=num_classes,
            dim=256,
            depth=12,
            heads=8,
            bucket_size=5,
            n_hashes=1,
            ff_mult=4, 
            lsh_dropout=0.1,  
            ff_dropout=0.1,  
            emb_dropout=0.1,
            use_rezero=False  
        )
    elif model_type == "vit":
        from vit_pytorch import ViT
        model = ViT(
            image_size=224,
            patch_size=8,
            num_classes=num_classes,
            dim=256,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif model_type == "vitface":
        from reformer.vit_pytorch import FaceTransformer  
        model = FaceTransformer(
            num_classes=num_classes,
            arc_s=4.0, 
            arc_m=0.23
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model.to(device)


def train(args):
    # Initialize TensorBoard
    writer = SummaryWriter(args.tensorboard_dir)

    # Create checkpoint directory
    checkpoint_dir = args.model_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Dataset selection and configuration
    print(f"Load data loader from {args.data_dir}")
    if args.data_dir is not None:
        train_loader, test_loader = get_student_dataloader(args.data_dir, batch_size=args.batch_size)
        
        all_labels = []
        for dataset in train_loader.dataset.datasets:
            all_labels.extend(dataset.labels)

        # Calculate the number of unique classes
        num_classes = len(set(all_labels))
        print(f"Number of classes: {num_classes}")
        
    else:
        train_loader, test_loader = get_lpfw_dataloaders(args.batch_size)
        num_classes = len(train_loader.dataset.dataset.class_to_idx)
        
    # Initialize model
    model = initialize_model(args.model_type, num_classes, device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load checkpoint if it exists
    start_epoch = 0
    best_accuracy = 0.0
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        print(f"Resuming training from epoch {start_epoch} with best accuracy {best_accuracy:.4f}")

    # Training loop
    num_epochs = args.epochs
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, device)
        test_accuracy = evaluate(model, test_loader, criterion, epoch, writer, device)

        # Save checkpoint if test accuracy improves
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"New best accuracy: {best_accuracy:.4f}. Model saved.")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint.pth"))
            print(f"Checkpoint saved at epoch {epoch + 1}")

    # Save final model after all epochs
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")

    # Close TensorBoard writer
    writer.close()


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, device):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False)
    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate predictions
        _, preds = torch.max(logits, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        train_loader_tqdm.set_postfix({"Loss": f"{loss.item():.4f}"})

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"Train Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.4f}")
    writer.add_scalar("Train/Loss", total_loss / len(train_loader), epoch)
    writer.add_scalar("Train/Accuracy", acc, epoch)
    writer.add_scalar("Train/Precision", precision, epoch)
    writer.add_scalar("Train/Recall", recall, epoch)
    writer.add_scalar("Train/F1", f1, epoch)

    return acc


def evaluate(model, test_loader, criterion, epoch, writer, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Calculate predictions
            _, preds = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"Test Loss: {total_loss / len(test_loader):.4f}, Accuracy: {acc:.4f}")
    writer.add_scalar("Test/Loss", total_loss / len(test_loader), epoch)
    writer.add_scalar("Test/Accuracy", acc, epoch)
    writer.add_scalar("Test/Precision", precision, epoch)
    writer.add_scalar("Test/Recall", recall, epoch)
    writer.add_scalar("Test/F1", f1, epoch)

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with ArcMargin")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--tensorboard-dir", type=str, default="/opt/ml/output/tensorboard/", help="Directory for TensorBoard logs")
    parser.add_argument("--model-dir", type=str, default="./models", help="Directory to save trained models")
    parser.add_argument("--model-type", type=str, default="vir", choices=["vir", "vit", "virface", "vitface"], help="Model type to train (vir or vit)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory containing the dataset")

    args = parser.parse_args()
    train(args)