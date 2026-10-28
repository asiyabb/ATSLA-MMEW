import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
from dataset import build_mmew_records, VideoDataset
from model import DFERmodel
from helper import load_pretrained_weights

def evaluate_and_visualize():
    records, class_names = build_mmew_records()
    unique_subjects = sorted(list(set([r[4] for r in records])))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_preds = []
    all_targets = []
    
    # Iterate through folds to collect predictions across all test splits
    for fold, (train_sub_indices, test_sub_indices) in enumerate(kf.split(unique_subjects), 1):
        test_subs = set([unique_subjects[i] for i in test_sub_indices])
        test_records = [r for r in records if r[4] in test_subs]
        
        test_data = VideoDataset(test_records, mode='test', image_size=112)
        val_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)
        
        model = DFERmodel(num_classes=len(class_names)).cuda()
        
        # Note: If you saved fold-specific models, you can load them here. 
        # For demonstration, we check for a general or last fold model checkpoint.
        checkpoint_path = f"best_atsla_mmew_fold{fold}.pth" if os.path.exists(f"best_atsla_mmew_fold{fold}.pth") else "best_atsla_mmew.pth"
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cuda')
            model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.cuda(), targets.cuda()
                output, _ = model(images)
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    war = 100.0 * np.mean(all_preds == all_targets)
    recalls = [np.mean(all_preds[all_targets == c] == c) for c in range(len(class_names)) if np.sum(all_targets == c) > 0]
    uar = 100.0 * np.mean(recalls) if recalls else 0.0
    
    print("\n==================== 5-FOLD AGGREGATED EVALUATION REPORT ====================")
    print(f"Overall Aggregated WAR (Accuracy): {war:.2f}%")
    print(f"Overall Aggregated UAR (Recall): {uar:.2f}%")
    print("-----------------------------------------------------------------------------")
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))
    print("=============================================================================\n")
    
    # Plot Confusion Matrix using Matplotlib
    print("Generating Confusion Matrix visualization (cm.png)...")
    cm = confusion_matrix(all_targets, all_preds)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6) * 100
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix - 5-Fold MMEW Macro-Expression (%)')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    thresh = cm_normalized.max() / 2.0
    for i, j in np.ndindex(cm_normalized.shape):
        plt.text(j, i, f"{cm_normalized[i, j]:.1f}%",
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")
                 
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('cm.png', dpi=300)
    plt.close()
    print("-> Saved confusion matrix plot to 'cm.png'.")

if __name__ == '__main__':
    evaluate_and_visualize()