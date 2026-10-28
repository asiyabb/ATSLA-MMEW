import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import random
from sklearn.model_selection import KFold
from torch.amp import autocast, GradScaler
from dataset import build_mmew_records, VideoDataset
from model import DFERmodel, CombinedEmotionLoss
from helper import load_pretrained_weights

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--img-size', type=int, default=112)
    parser.add_argument('--pretrain-path', type=str, default='Pretrain_DFEW_MAFW.pth')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    records, class_names = build_mmew_records()
    
    # Extract unique subjects for subject-independent splitting
    unique_subjects = sorted(list(set([r[4] for r in records])))
    print(f"Total unique subjects found: {len(unique_subjects)}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_wars = []
    fold_uars = []
    
    for fold, (train_sub_indices, test_sub_indices) in enumerate(kf.split(unique_subjects), 1):
        print(f"\n==================== FOLD {fold}/5 ====================")
        train_subs = set([unique_subjects[i] for i in train_sub_indices])
        test_subs = set([unique_subjects[i] for i in test_sub_indices])
        
        train_records = [r for r in records if r[4] in train_subs]
        test_records = [r for r in records if r[4] in test_subs]
        
        print(f"Train subjects: {len(train_subs)} | Test subjects: {len(test_subs)}")
        print(f"Train samples: {len(train_records)} | Test samples: {len(test_records)}")
        
        # Compute class weights for training split
        train_labels = [r[2] for r in train_records]
        class_counts = np.bincount(train_labels, minlength=len(class_names))
        total_samples = sum(class_counts)
        class_weights = torch.FloatTensor([total_samples / (len(class_names) * c) if c > 0 else 1.0 for c in class_counts]).cuda()
        
        train_data = VideoDataset(train_records, mode='train', image_size=args.img_size)
        test_data = VideoDataset(test_records, mode='test', image_size=args.img_size)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        model = DFERmodel(num_classes=len(class_names)).cuda()
        
        if os.path.exists(args.pretrain_path):
            checkpoint = torch.load(args.pretrain_path, map_location='cuda')
            model = load_pretrained_weights(model, checkpoint)
            
        criterion = CombinedEmotionLoss(class_weights=class_weights, num_classes=len(class_names)).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
        scaler = GradScaler('cuda')
        
        best_fold_war = 0.0
        for epoch in range(args.epochs):
            model.train()
            for images, targets, video_ids in train_loader:
                images, targets = images.cuda(), targets.cuda()
                b, t, c, h, w = images.shape
                
                with autocast('cuda'):
                    output, frame_features = model(images)
                    loss, _ = criterion(frame_features, output, targets, video_ids, b, t)
                    
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            scheduler.step()
            criterion.reset_memory()
            
            # Validation
            model.eval()
            preds, gts = [], []
            with torch.no_grad():
                for images, targets, video_ids in val_loader:
                    images, targets = images.cuda(), targets.cuda()
                    output, _ = model(images)
                    preds.extend(output.argmax(dim=1).cpu().numpy())
                    gts.extend(targets.cpu().numpy())
                    
            war = 100.0 * np.mean(np.array(preds) == np.array(gts))
            if war > best_fold_war:
                best_fold_war = war
                
        print(f"--> Fold {fold} Best WAR: {best_fold_war:.2f}%")
        fold_wars.append(best_fold_war)
        
    print("\n==================== CROSS-VALIDATION SUMMARY ====================")
    print(f"Average WAR across 5 folds: {np.mean(fold_wars):.2f}% (± {np.std(fold_wars):.2f}%)")
    print("==================================================================")

if __name__ == '__main__':
    main()