"""
Verification script to analyze Forensics Adapter test results in detail.
This script helps you understand if your model is working correctly.
"""

import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_predictions(predictions, labels, save_dir='analysis_results'):
    """
    Comprehensive analysis of model predictions.
    
    Args:
        predictions: numpy array of prediction scores (0-1)
        labels: numpy array of ground truth labels (0=real, 1=fake)
        save_dir: directory to save analysis plots
    """
    
    # Create save directory
    Path(save_dir).mkdir(exist_ok=True)
    
    print("="*70)
    print("FORENSICS ADAPTER - PREDICTION ANALYSIS")
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
    
    print(f"\nReal samples - Mean prediction: {real_preds.mean():.4f} (should be LOW)")
    print(f"Fake samples - Mean prediction: {fake_preds.mean():.4f} (should be HIGH)")
    
    # Frame-level metrics
    print("\n2. FRAME-LEVEL METRICS")
    print("-"*70)
    
    # ROC curve and AUC
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    
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
    precision = metrics.precision_score(labels, pred_binary)
    recall = metrics.recall_score(labels, pred_binary)
    f1 = metrics.f1_score(labels, pred_binary)
    
    print(f"AUC: {auc:.4f}")
    print(f"EER: {eer:.4f} (at threshold={eer_threshold:.4f})")
    print(f"Accuracy @ 0.5: {acc_50:.4f}")
    print(f"Accuracy @ EER: {acc_eer:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion matrix at threshold=0.5
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix (threshold=0.5):")
    print(f"  True Negatives (Real correctly classified):  {tn}")
    print(f"  False Positives (Real classified as Fake):   {fp}")
    print(f"  False Negatives (Fake classified as Real):   {fn}")
    print(f"  True Positives (Fake correctly classified):  {tp}")
    
    # Video-level analysis (assuming 32 frames per video)
    print("\n3. VIDEO-LEVEL ANALYSIS")
    print("-"*70)
    
    frames_per_video = 32
    num_videos = len(predictions) // frames_per_video
    
    if len(predictions) % frames_per_video == 0:
        print(f"Detected {num_videos} videos ({frames_per_video} frames each)")
        print("\nNOTE: This assumes frames are in order. Actual test shuffles frames!")
        print("      Real video-level metric groups by parsing image paths.\n")
        
        for i in range(num_videos):
            start = i * frames_per_video
            end = (i + 1) * frames_per_video
            
            video_preds = predictions[start:end]
            video_labels = labels[start:end]
            
            avg_pred = video_preds.mean()
            true_label = int(video_labels.mean())
            label_str = "FAKE" if true_label == 1 else "REAL"
            
            frame_acc = np.mean((video_preds > 0.5).astype(int) == video_labels)
            
            print(f"Video {i}: True={label_str}, AvgScore={avg_pred:.4f}, FrameAcc={frame_acc:.1%}")
    
    # Prediction distribution analysis
    print("\n4. PREDICTION DISTRIBUTION")
    print("-"*70)
    
    # Count predictions in ranges
    very_low = np.sum(predictions < 0.2)
    low = np.sum((predictions >= 0.2) & (predictions < 0.4))
    mid = np.sum((predictions >= 0.4) & (predictions < 0.6))
    high = np.sum((predictions >= 0.6) & (predictions < 0.8))
    very_high = np.sum(predictions >= 0.8)
    
    print(f"Very Low (<0.2): {very_low:3d} samples ({very_low/len(predictions)*100:.1f}%)")
    print(f"Low (0.2-0.4):   {low:3d} samples ({low/len(predictions)*100:.1f}%)")
    print(f"Medium (0.4-0.6):{mid:3d} samples ({mid/len(predictions)*100:.1f}%)")
    print(f"High (0.6-0.8):  {high:3d} samples ({high/len(predictions)*100:.1f}%)")
    print(f"Very High (>0.8):{very_high:3d} samples ({very_high/len(predictions)*100:.1f}%)")
    
    # Model confidence
    print("\n5. MODEL CONFIDENCE")
    print("-"*70)
    
    # Calculate confidence (distance from 0.5)
    confidence = np.abs(predictions - 0.5)
    avg_confidence = confidence.mean()
    
    print(f"Average confidence: {avg_confidence:.4f}")
    print(f"  (0.0 = uncertain, 0.5 = very confident)")
    
    confident_samples = np.sum(confidence > 0.3)
    print(f"Confident predictions (|pred-0.5|>0.3): {confident_samples}/{len(predictions)} ({confident_samples/len(predictions)*100:.1f}%)")
    
    # Errors analysis
    print("\n6. ERROR ANALYSIS")
    print("-"*70)
    
    errors = pred_binary != labels
    error_indices = np.where(errors)[0]
    
    print(f"Total errors: {errors.sum()}/{len(predictions)} ({errors.mean()*100:.1f}%)")
    
    if errors.sum() > 0:
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
    
    # Visualizations
    print("\n7. GENERATING VISUALIZATIONS")
    print("-"*70)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
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
    ax3.axhline(max(accuracies), color='orange', linestyle=':', linewidth=1, label=f'Max ({max(accuracies):.3f})')
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Accuracy vs Threshold', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix Heatmap
    ax4 = plt.subplot(2, 3, 4)
    im = ax4.imshow(cm, cmap='Blues', aspect='auto')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Real', 'Fake'], fontsize=12)
    ax4.set_yticklabels(['Real', 'Fake'], fontsize=12)
    ax4.set_xlabel('Predicted', fontsize=12)
    ax4.set_ylabel('True', fontsize=12)
    ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add text annotations
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
    save_path = Path(save_dir) / 'prediction_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {save_path}")
    
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ AUC: {auc:.4f} {'(Good)' if auc > 0.75 else '(Needs improvement)'}")
    print(f"✓ Accuracy: {acc_50:.4f} {'(Good)' if acc_50 > 0.55 else '(Needs improvement)'}")
    print(f"✓ Separation: Real avg={real_preds.mean():.3f}, Fake avg={fake_preds.mean():.3f}")
    
    separation = fake_preds.mean() - real_preds.mean()
    if separation > 0.2:
        print(f"✓ Model shows good separation between real and fake! (Δ={separation:.3f})")
    elif separation > 0:
        print(f"⚠ Model shows weak separation (Δ={separation:.3f})")
    else:
        print(f"✗ Model is not separating real from fake well (Δ={separation:.3f})")
    
    print("\n" + "="*70)
    print("CONCLUSION: Your model is", end=" ")
    if auc > 0.75 and separation > 0.1:
        print("WORKING WELL! ✅")
    elif auc > 0.6:
        print("WORKING MODERATELY ⚠️")
    else:
        print("NEEDS ATTENTION ❌")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FORENSICS ADAPTER - RESULT VERIFICATION TOOL")
    print("="*70)
    print("\nThis script analyzes your test results in detail.")
    print("\nUsage:")
    print("  1. Copy your predictions and labels from test.py output")
    print("  2. Paste them below (replace the example data)")
    print("  3. Run this script to get comprehensive analysis\n")
    
    # Updated with 20 videos (10 real + 10 fake) results - November 18, 2025
    predictions = np.array([0.32514915, 0.9224444, 0.68776196, 0.9626228, 0.7324297, 0.9991522,
 0.23520173, 0.18102041, 0.5737838, 0.993216, 0.99376136, 0.12252504,
 0.6593729, 0.89402616, 0.30530143, 0.26911315, 0.80834067, 0.2831713,
 0.99421316, 0.3273566, 0.95964915, 0.9698169, 0.9983758, 0.99784935,
 0.9908709, 0.9936446, 0.9444173, 0.9987369, 0.99185705, 0.55408436,
 0.9917629, 0.95652723, 0.3664247, 0.9997385, 0.9965352, 0.1801034,
 0.9971521, 0.85901976, 0.9412769, 0.99967766, 0.9978144, 0.6881363,
 0.3541316, 0.94538635, 0.5026959, 0.48462301, 0.09680269, 0.27136657,
 0.2900614, 0.3095336, 0.59911567, 0.9954058, 0.86833984, 0.6467714,
 0.44846076, 0.9672527, 0.98560786, 0.99306124, 0.9621756, 0.37541562,
 0.34879035, 0.31248665, 0.21512605, 0.99539787, 0.27953556, 0.36257222,
 0.67030096, 0.75954306, 0.25025845, 0.9291101, 0.46178117, 0.9998574,
 0.9917173, 0.46100533, 0.92397344, 0.46540803, 0.31548512, 0.9649273,
 0.2393266, 0.98606306, 0.5161808, 0.9875339, 0.49856868, 0.89434814,
 0.33221093, 0.99575293, 0.9846581, 0.8032, 0.9998883, 0.8723956,
 0.60494906, 0.3896669, 0.93821776, 0.5245455, 0.47379732, 0.9974727,
 0.36626068, 0.23395015, 0.5657419, 0.8619375, 0.36939296, 0.43755645,
 0.45318404, 0.24660265, 0.98802173, 0.847677, 0.13422516, 0.21874778,
 0.98568845, 0.9864401, 0.9557516, 0.41331053, 0.5109371, 0.65045744,
 0.99893254, 0.39378032, 0.99955624, 0.64711225, 0.55991495, 0.9698047,
 0.9997944, 0.33012074, 0.5112346, 0.36699328, 0.99540615, 0.34037304,
 0.18531306, 0.2524291, 0.45899564, 0.6380808, 0.5387374, 0.9080406,
 0.99896705, 0.9998437, 0.12121259, 0.5073825, 0.2423873, 0.99720913,
 0.91322297, 0.9976406, 0.47388974, 0.40734154, 0.9648301, 0.44557786,
 0.41027415, 0.99839383, 0.98473144, 0.98225135, 0.99974173, 0.44933954,
 0.37525612, 0.46866426, 0.7640667, 0.28514767, 0.6677719, 0.3389787,
 0.22797364, 0.31495067, 0.31843287, 0.28096917, 0.99976593, 0.8673829,
 0.9992409, 0.4758485, 0.9998254, 0.93881303, 0.9945103, 0.6086845,
 0.99054587, 0.30263954, 0.6780666, 0.99937564, 0.23078495, 0.25461227,
 0.7992271, 0.9228361, 0.95741916, 0.99512273, 0.15307456, 0.9927769,
 0.6880089, 0.99272776, 0.40144786, 0.87587136, 0.9311666, 0.20481282,
 0.563733, 0.2730396, 0.68157464, 0.41991192, 0.9994967, 0.8398192,
 0.6124194, 0.9850631, 0.2543978, 0.5190106, 0.9563824, 0.42488503,
 0.76499844, 0.26394978, 0.9959906, 0.9668308, 0.42939743, 0.26691973,
 0.46528152, 0.39557627, 0.7759924, 0.875185, 0.8849891, 0.74170667,
 0.95286876, 0.39091954, 0.22567177, 0.8676639, 0.99993515, 0.8061261,
 0.99222094, 0.33110863, 0.99940753, 0.8371374, 0.93068475, 0.27100292,
 0.7024242, 0.9074001, 0.75890577, 0.51332325, 0.7337206, 0.9858064,
 0.23119344, 0.8368911, 0.5820585, 0.15010941, 0.50026035, 0.9955954,
 0.9855041, 0.9820857, 0.99883884, 0.40163445, 0.9443399, 0.475102,
 0.24220008, 0.16991657, 0.36509693, 0.89509064, 0.2930169, 0.23077588,
 0.3157467, 0.9985026, 0.95247597, 0.7867169, 0.98629826, 0.38675046,
 0.7857611, 0.2152749, 0.45045027, 0.58357304, 0.3966436, 0.5751864,
 0.48124298, 0.25646976, 0.38006416, 0.4835362, 0.9952377, 0.24846628,
 0.9969625, 0.98821646, 0.8555223, 0.9734754, 0.93077534, 0.33435026,
 0.99197245, 0.9661412, 0.34587848, 0.42529246, 0.68401575, 0.95910746,
 0.92362624, 0.45973158, 0.9066062, 0.37375104, 0.2448038, 0.96006495,
 0.45879576, 0.9900283, 0.32663223, 0.6259124, 0.71101713, 0.45674157,
 0.99724007, 0.9354998, 0.9886645, 0.9956239, 0.9297779, 0.9344781,
 0.9932334, 0.6048423, 0.37427223, 0.6011672, 0.29026422, 0.78232193,
 0.7272615, 0.99994135, 0.63902825, 0.99514234, 0.40010372, 0.4748062,
 0.27341184, 0.99922466, 0.9904276, 0.523457, 0.9712167, 0.9117524,
 0.9954254, 0.9491858, 0.52364045, 0.99673814, 0.6153485, 0.9911555,
 0.9698826, 0.45493436, 0.99858165, 0.21488069, 0.8473016, 0.93303686,
 0.7489185, 0.99625325, 0.53189546, 0.9990559, 0.99823296, 0.9995328,
 0.8624458, 0.4964051, 0.9814761, 0.4627176, 0.8162076, 0.9587291,
 0.490663, 0.26472312, 0.09268796, 0.9943581, 0.99918085, 0.99477166,
 0.29159296, 0.99682546, 0.9207339, 0.99690527, 0.19693002, 0.38532758,
 0.54374254, 0.2727335, 0.35878167, 0.9831955, 0.9954028, 0.67009777,
 0.38187477, 0.877755, 0.7502364, 0.20837416, 0.787579, 0.99944025,
 0.27953908, 0.15292363, 0.99936885, 0.43678933, 0.99964535, 0.99982613,
 0.7446923, 0.9962896, 0.36599305, 0.98886055, 0.97354275, 0.3499055,
 0.4180625, 0.845054, 0.99869055, 0.99455476, 0.27553138, 0.6765301,
 0.6440928, 0.5596377, 0.5465662, 0.9975611, 0.99943715, 0.30025217,
 0.99266, 0.40446752, 0.99970394, 0.3162728, 0.25874114, 0.22216126,
 0.9192611, 0.45642766, 0.99717206, 0.9293473, 0.97082704, 0.5332444,
 0.4191137, 0.45358047, 0.9940147, 0.594951, 0.1544346, 0.90416104,
 0.9690624, 0.15897803, 0.69454515, 0.9845957, 0.34413412, 0.2426199,
 0.9925323, 0.774701, 0.99979204, 0.98612034, 0.99985814, 0.9999393,
 0.48866978, 0.61402065, 0.99606234, 0.7424499, 0.9689754, 0.9821023,
 0.76865494, 0.7255376, 0.9468751, 0.94532, 0.6315485, 0.3139205,
 0.99972767, 0.827786, 0.5632935, 0.99510443, 0.9999094, 0.45305464,
 0.88259095, 0.87541294, 0.33478805, 0.4953777, 0.4023601, 0.52805716,
 0.22827673, 0.6358617, 0.9947404, 0.41796064, 0.99575824, 0.9972568,
 0.99870515, 0.99486923, 0.83235735, 0.9808383, 0.9762896, 0.5977264,
 0.96375185, 0.9201484, 0.99181306, 0.4969864, 0.99853337, 0.9992224,
 0.7646776, 0.26496106, 0.80363303, 0.8698466, 0.98142403, 0.29187557,
 0.12253784, 0.68254673, 0.21419144, 0.99979526, 0.9145717, 0.9799675,
 0.90180796, 0.99462056, 0.28233355, 0.47640672, 0.9723551, 0.9992588,
 0.28012165, 0.35081315, 0.46796596, 0.2432918, 0.5972581, 0.986691,
 0.6810254, 0.584846, 0.35635304, 0.99096876, 0.27049235, 0.99991965,
 0.98967886, 0.99400854, 0.9993144, 0.73800886, 0.4999279, 0.9799397,
 0.99981767, 0.90901196, 0.9516095, 0.9863239, 0.73641163, 0.7435629,
 0.55319345, 0.9745359, 0.98579377, 0.8038756, 0.15389842, 0.5941285,
 0.29826504, 0.5517124, 0.9999193, 0.28680438, 0.56806344, 0.99657696,
 0.9986873, 0.9505101, 0.27118987, 0.9901322, 0.2130807, 0.99435353,
 0.27429044, 0.90547174, 0.96969527, 0.37020624, 0.97838986, 0.13916676,
 0.98354983, 0.9974183, 0.99970835, 0.9357895, 0.9973092, 0.8824099,
 0.6873444, 0.5294129, 0.9968669, 0.9945427, 0.53430516, 0.8143534,
 0.9979961, 0.26191324, 0.23794287, 0.44046187, 0.46034887, 0.12859032,
 0.99564004, 0.99843735, 0.7198034, 0.23747316, 0.9912918, 0.17576788,
 0.80870175, 0.789548, 0.74291784, 0.8658579, 0.39847383, 0.46111426,
 0.9873103, 0.81867236, 0.9997181, 0.8433997, 0.21140355, 0.99981815,
 0.34363416, 0.9998983, 0.99793845, 0.728772, 0.43071994, 0.12455994,
 0.9957883, 0.40652046, 0.97800696, 0.992147, 0.540708, 0.6174759,
 0.9882703, 0.6620133, 0.18825595, 0.97783047, 0.3993417, 0.93193614,
 0.995479, 0.9896108, 0.2445473, 0.99988544, 0.60858923, 0.99056315,
 0.9807694, 0.99649316, 0.4334436, 0.99953616, 0.9400383, 0.21797207,
 0.12234915, 0.99831426, 0.99656457, 0.51640964, 0.9865935, 0.6052038,
 0.48142123, 0.9932284, 0.74106103, 0.41525006, 0.35096946, 0.43818444,
 0.7977365, 0.5566571, 0.9987282, 0.90410304, 0.5935593, 0.9816828,
 0.25669238, 0.9995751, 0.9681635, 0.9941414, 0.36623806, 0.17273584,
 0.34272167, 0.9775844, 0.9789929, 0.36890537, 0.93798727, 0.3589284,
 0.38720515, 0.58678657, 0.4490283, 0.97199935, 0.16721767, 0.39118102,
 0.999864, 0.99960965, 0.9192456, 0.96605474, 0.5048849, 0.98692894,
 0.807114, 0.45937142, 0.37413043, 0.9601957, 0.7631985, 0.20786245,
 0.7556019, 0.48227492])
    
    labels = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1,
 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1,
 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1,
 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1,
 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0,
 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0,
 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1,
 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0,
 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1,
 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1,
 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1,
 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0,
 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0,
 1, 1, 0, 0, 1, 1, 1, 0, 0])
    
    # Run analysis
    analyze_predictions(predictions, labels)
    
    print("\n✓ Analysis complete!")
    print("  Check the 'analysis_results' folder for visualizations.")

