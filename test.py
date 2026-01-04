"""
eval pretained model.
"""
import numpy as np
import random
import yaml
import os
from tqdm import tqdm
from pathlib import Path
from trainer.metrics.utils import get_test_metrics
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from sklearn import metrics as sklearn_metrics
from sklearn.metrics import confusion_matrix

# Use non-interactive backend for matplotlib (works without display)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from model.ds import DS

import argparse

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, default='config/test.yaml')
parser.add_argument('--weights_path', type=str, default='weights/ckpt_best.pth')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument("--analyze", action='store_true', default=True,
                    help="Run detailed analysis and generate visualizations (default: True)")
parser.add_argument("--no-analyze", action='store_false', dest='analyze',
                    help="Skip detailed analysis")
parser.add_argument("--output-dir", type=str, default='analysis_results',
                    help="Directory to save analysis results (default: analysis_results)")

args = parser.parse_args()

device = torch.device("cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    #feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        #feature_lists += list(predictions['feat'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists)#,np.array(feature_lists)
    
def analyze_predictions(predictions, labels, dataset_name, save_dir='analysis_results'):
    """
    Comprehensive analysis of model predictions with visualizations.
    
    Args:
        predictions: numpy array of prediction scores (0-1)
        labels: numpy array of ground truth labels (0=real, 1=fake)
        dataset_name: name of the dataset being analyzed
        save_dir: directory to save analysis plots
    """
    # Create save directory
    Path(save_dir).mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print(f"DETAILED ANALYSIS: {dataset_name}")
    print("="*70)
    
    # Basic statistics
    print("\n1. PREDICTION STATISTICS")
    print("-"*70)
    print(f"Total samples: {len(predictions)}")
    print(f"Real samples (label=0): {np.sum(labels == 0)}")
    print(f"Fake samples (label=1): {np.sum(labels == 1)}")
    print(f"\nPrediction score range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Prediction mean: {predictions.mean():.4f}")
    print(f"Prediction std: {predictions.std():.4f}")
    
    # Separate by label
    real_preds = predictions[labels == 0]
    fake_preds = predictions[labels == 1]
    
    print(f"\nReal samples - Mean prediction: {real_preds.mean():.4f} (should be LOW, close to 0)")
    print(f"Fake samples - Mean prediction: {fake_preds.mean():.4f} (should be HIGH, close to 1)")
    
    # Frame-level metrics
    print("\n2. FRAME-LEVEL METRICS")
    print("-"*70)
    
    # ROC curve and AUC
    fpr, tpr, thresholds = sklearn_metrics.roc_curve(labels, predictions, pos_label=1)
    auc = sklearn_metrics.auc(fpr, tpr)
    
    # EER
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    # Accuracy at different thresholds
    acc_50 = np.mean((predictions > 0.5).astype(int) == labels)
    acc_eer = np.mean((predictions > eer_threshold).astype(int) == labels)
    
    # Precision, Recall, F1
    pred_binary = (predictions > 0.5).astype(int)
    precision = sklearn_metrics.precision_score(labels, pred_binary, zero_division=0)
    recall = sklearn_metrics.recall_score(labels, pred_binary, zero_division=0)
    f1 = sklearn_metrics.f1_score(labels, pred_binary, zero_division=0)
    
    print(f"AUC: {auc:.4f}")
    print(f"EER: {eer:.4f} (at threshold={eer_threshold:.4f})")
    print(f"Accuracy @ 0.5: {acc_50:.4f}")
    print(f"Accuracy @ EER threshold: {acc_eer:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion matrix at threshold=0.5
    cm = confusion_matrix(labels, pred_binary)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix (threshold=0.5):")
        print(f"  True Negatives (Real correctly classified):  {tn}")
        print(f"  False Positives (Real classified as Fake):   {fp}")
        print(f"  False Negatives (Fake classified as Real):   {fn}")
        print(f"  True Positives (Fake correctly classified):  {tp}")
    
    # Prediction distribution analysis
    print("\n3. PREDICTION DISTRIBUTION")
    print("-"*70)
    
    # Count predictions in ranges
    very_low = np.sum(predictions < 0.2)
    low = np.sum((predictions >= 0.2) & (predictions < 0.4))
    mid = np.sum((predictions >= 0.4) & (predictions < 0.6))
    high = np.sum((predictions >= 0.6) & (predictions < 0.8))
    very_high = np.sum(predictions >= 0.8)
    
    print(f"Very Low (<0.2):  {very_low:3d} samples ({very_low/len(predictions)*100:.1f}%) - Confident REAL")
    print(f"Low (0.2-0.4):    {low:3d} samples ({low/len(predictions)*100:.1f}%) - Leaning REAL")
    print(f"Medium (0.4-0.6): {mid:3d} samples ({mid/len(predictions)*100:.1f}%) - Uncertain")
    print(f"High (0.6-0.8):   {high:3d} samples ({high/len(predictions)*100:.1f}%) - Leaning FAKE")
    print(f"Very High (>0.8): {very_high:3d} samples ({very_high/len(predictions)*100:.1f}%) - Confident FAKE")
    
    # Model confidence
    print("\n4. MODEL CONFIDENCE")
    print("-"*70)
    
    confidence = np.abs(predictions - 0.5)
    avg_confidence = confidence.mean()
    
    print(f"Average confidence: {avg_confidence:.4f} (0.0=uncertain, 0.5=very confident)")
    
    confident_samples = np.sum(confidence > 0.3)
    print(f"Confident predictions (|pred-0.5|>0.3): {confident_samples}/{len(predictions)} ({confident_samples/len(predictions)*100:.1f}%)")
    
    # Error analysis
    print("\n5. ERROR ANALYSIS")
    print("-"*70)
    
    errors = pred_binary != labels
    print(f"Total errors: {errors.sum()}/{len(predictions)} ({errors.mean()*100:.1f}%)")
    
    # False positives (real predicted as fake)
    fp_indices = np.where((labels == 0) & (predictions > 0.5))[0]
    print(f"\nFalse Positives (real→fake): {len(fp_indices)}")
    if len(fp_indices) > 0:
        fp_scores = predictions[fp_indices]
        print(f"  Score range: [{fp_scores.min():.4f}, {fp_scores.max():.4f}]")
        print(f"  Average score: {fp_scores.mean():.4f}")
    
    # False negatives (fake predicted as real)
    fn_indices = np.where((labels == 1) & (predictions <= 0.5))[0]
    print(f"\nFalse Negatives (fake→real): {len(fn_indices)}")
    if len(fn_indices) > 0:
        fn_scores = predictions[fn_indices]
        print(f"  Score range: [{fn_scores.min():.4f}, {fn_scores.max():.4f}]")
        print(f"  Average score: {fn_scores.mean():.4f}")
    
    # Generate visualizations
    print("\n6. GENERATING VISUALIZATIONS")
    print("-"*70)
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Analysis: {dataset_name}', fontsize=16, fontweight='bold')
    
    # 1. ROC Curve
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    ax1.plot(fpr[eer_idx], tpr[eer_idx], 'go', markersize=10, label=f'EER = {eer:.4f}')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Prediction Distribution by Label
    ax2 = plt.subplot(2, 3, 2)
    bins = np.linspace(0, 1, 30)
    ax2.hist(real_preds, bins=bins, alpha=0.6, label='Real (label=0)', color='green', edgecolor='black')
    ax2.hist(fake_preds, bins=bins, alpha=0.6, label='Fake (label=1)', color='red', edgecolor='black')
    ax2.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
    ax2.set_xlabel('Prediction Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Prediction Distribution by Label', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Threshold vs Accuracy
    ax3 = plt.subplot(2, 3, 3)
    thresholds_test = np.linspace(0, 1, 100)
    accuracies = []
    for thresh in thresholds_test:
        pred_thresh = (predictions > thresh).astype(int)
        acc = np.mean(pred_thresh == labels)
        accuracies.append(acc)
    
    ax3.plot(thresholds_test, accuracies, 'b-', linewidth=2)
    ax3.axvline(0.5, color='red', linestyle='--', linewidth=1, label='Default (0.5)')
    ax3.axvline(eer_threshold, color='green', linestyle='--', linewidth=1, label=f'EER ({eer_threshold:.3f})')
    best_thresh_idx = np.argmax(accuracies)
    ax3.axhline(max(accuracies), color='orange', linestyle=':', linewidth=1, 
                label=f'Max ({max(accuracies):.3f} @ {thresholds_test[best_thresh_idx]:.2f})')
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Accuracy vs Threshold', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix Heatmap
    ax4 = plt.subplot(2, 3, 4)
    if cm.size == 4:
        im = ax4.imshow(cm, cmap='Blues', aspect='auto')
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(['Real', 'Fake'], fontsize=12)
        ax4.set_yticklabels(['Real', 'Fake'], fontsize=12)
        ax4.set_xlabel('Predicted', fontsize=12)
        ax4.set_ylabel('True', fontsize=12)
        ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        for i in range(2):
            for j in range(2):
                text = ax4.text(j, i, cm[i, j], ha="center", va="center", 
                              color="white" if cm[i, j] > cm.max()/2 else "black",
                              fontsize=16, fontweight='bold')
        plt.colorbar(im, ax=ax4, fraction=0.046)
    
    # 5. Prediction scatter plot
    ax5 = plt.subplot(2, 3, 5)
    indices = np.arange(len(predictions))
    real_idx = labels == 0
    fake_idx = labels == 1
    ax5.scatter(indices[real_idx], predictions[real_idx], c='green', alpha=0.5, 
               s=30, label='Real', edgecolors='black', linewidths=0.5)
    ax5.scatter(indices[fake_idx], predictions[fake_idx], c='red', alpha=0.5, 
               s=30, label='Fake', edgecolors='black', linewidths=0.5)
    ax5.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax5.set_xlabel('Frame Index', fontsize=12)
    ax5.set_ylabel('Prediction Score', fontsize=12)
    ax5.set_title('Predictions by Frame', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Box plot
    ax6 = plt.subplot(2, 3, 6)
    box_data = [real_preds, fake_preds]
    bp = ax6.boxplot(box_data, labels=['Real', 'Fake'], patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.6)
    ax6.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax6.set_ylabel('Prediction Score', fontsize=12)
    ax6.set_title('Prediction Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_dir) / f'{dataset_name}_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {save_path}")
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    separation = fake_preds.mean() - real_preds.mean()
    
    print(f"✓ AUC: {auc:.4f} {'(Excellent)' if auc > 0.9 else '(Good)' if auc > 0.75 else '(Needs improvement)'}")
    print(f"✓ Accuracy: {acc_50:.4f} {'(Good)' if acc_50 > 0.8 else '(Moderate)' if acc_50 > 0.6 else '(Low)'}")
    print(f"✓ Separation: Real avg={real_preds.mean():.3f}, Fake avg={fake_preds.mean():.3f}, Δ={separation:.3f}")
    
    if separation > 0.3:
        print(f"✓ Model shows STRONG separation between real and fake!")
    elif separation > 0.1:
        print(f"⚠ Model shows moderate separation")
    elif separation > 0:
        print(f"⚠ Model shows weak separation")
    else:
        print(f"✗ Model is NOT separating real from fake properly!")
    
    print("\nCONCLUSION: ", end="")
    if auc > 0.85 and separation > 0.2:
        print("Model is WORKING WELL on this dataset! ✅")
    elif auc > 0.7 and separation > 0.1:
        print("Model is working MODERATELY on this dataset ⚠️")
    else:
        print("Model may NOT generalize well to this dataset ❌")
    print("="*70)
    
    return {
        'auc': auc,
        'eer': eer,
        'accuracy': acc_50,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'separation': separation,
        'real_mean': real_preds.mean(),
        'fake_mean': fake_preds.mean()
    }


def test_epoch(model, test_data_loaders, run_analysis=True, output_dir='analysis_results'):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}
    analysis_results = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps = test_one_dataset(model, test_data_loaders[key])
        print(f'name {data_dict.keys()}')
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")
        
        # Run detailed analysis if enabled
        if run_analysis:
            try:
                analysis = analyze_predictions(
                    predictions_nps, 
                    label_nps, 
                    dataset_name=key,
                    save_dir=output_dir
                )
                analysis_results[key] = analysis
            except Exception as e:
                print(f"Warning: Could not run analysis for {key}: {e}")

    return metrics_all_datasets, analysis_results

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)

    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)

    model = DS(clip_name=config['clip_model_name'],
               adapter_vit_name=config['vit_name'],
               num_quires=config['num_quires'],
               fusion_map=config['fusion_map'],
               mlp_dim=config['mlp_dim'],
               mlp_out_dim=config['mlp_out_dim'],
               head_num=config['head_num'],
               device=config['device'])
    epoch = 0
    #weights_paths = [
    #                 '/data/cuixinjie/DsClip_L_V2/logs/ds_ _2024-09-25-14-26-04/test/avg/ckpt_best.pth',
    #                ]

    try:
        epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
    except:
        epoch = 0
    
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.to(device)

    print(f'===> Load {weights_path} done!')

    # start testing
    best_metric, analysis_results = test_epoch(
        model, 
        test_data_loaders,
        run_analysis=args.analyze,
        output_dir=args.output_dir
    )
    
    print('\n===> Test Done!')
    
    if args.analyze and analysis_results:
        print(f'\n📊 Analysis results saved to: {args.output_dir}/')
        print('   Open the PNG files to see detailed visualizations.')

if __name__ == '__main__':
    main()
