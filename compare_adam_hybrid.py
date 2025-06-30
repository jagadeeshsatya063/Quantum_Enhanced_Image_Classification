import numpy as np
import matplotlib.pyplot as plt

def compare_adam_hybrid():
    # Load results
    adam_train_losses = np.load('adam_train_losses.npy')
    adam_val_accuracies = np.load('adam_val_accuracies.npy')
    hybrid_train_losses = np.load('hybrid_train_losses.npy')
    hybrid_val_accuracies = np.load('hybrid_val_accuracies.npy')

    # Load F1, recall, confusion matrix
    adam_f1 = np.load('adam_f1_scores.npy')[0]
    adam_recall = np.load('adam_recall_scores.npy')[0]
    adam_cm = np.load('adam_confusion_matrices.npy')[0]
    hybrid_f1 = np.load('hybrid_f1_scores.npy')[0]
    hybrid_recall = np.load('hybrid_recall_scores.npy')[0]
    hybrid_cm = np.load('hybrid_confusion_matrices.npy')[0]

    epochs = range(1, len(adam_train_losses) + 1)

    plt.figure(figsize=(14,6))
    plt.subplot(1,3,1)
    plt.plot(epochs, adam_train_losses, 'b-o', label='Adam Train Loss')
    plt.plot(epochs, hybrid_train_losses, 'g-o', label='Hybrid Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1,3,2)
    plt.plot(epochs, adam_val_accuracies, 'r-s', label='Adam Val Acc')
    plt.plot(epochs, hybrid_val_accuracies, 'm-s', label='Hybrid Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # F1 and Recall bar plot
    plt.subplot(1,3,3)
    bar_width = 0.35
    metrics = ['F1 Score', 'Recall']
    adam_metrics = [adam_f1, adam_recall]
    hybrid_metrics = [hybrid_f1, hybrid_recall]
    x = np.arange(len(metrics))
    plt.bar(x, adam_metrics, width=bar_width, label='Adam', color='blue', alpha=0.7)
    plt.bar(x + bar_width, hybrid_metrics, width=bar_width, label='Hybrid', color='green', alpha=0.7)
    plt.xticks(x + bar_width/2, metrics)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.title('Final F1 & Recall Scores')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('adam_vs_hybrid_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Comparison plot saved as 'adam_vs_hybrid_comparison.png'")

    # Print F1, Recall, and Confusion Matrices
    print("\nFinal Metrics:")
    print(f"Adam - F1 Score: {adam_f1:.4f}, Recall: {adam_recall:.4f}")
    print(f"Hybrid - F1 Score: {hybrid_f1:.4f}, Recall: {hybrid_recall:.4f}")
    print("\nAdam Confusion Matrix:\n", adam_cm)
    print("\nHybrid Confusion Matrix:\n", hybrid_cm)

if __name__ == "__main__":
    compare_adam_hybrid() 