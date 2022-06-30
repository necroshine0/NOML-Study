import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

def draw_PR(rec_t, pr_t, rec_v, pr_v, ax=None):
    if ax is None:
        ax = plt.gca()
        
    ax.plot(rec_t, pr_t, color='royalblue', label='train')
    ax.plot(rec_v, pr_v, color='lime', label='valid')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(shadow=False, fontsize=14)
    return ax

def draw_ROC(fpr_t, tpr_t, fpr_v, tpr_v, ax=None):
    if ax is None:
        ax = plt.gca()
        
    ax.plot(fpr_t, tpr_t, color='royalblue', label='train')
    ax.plot(fpr_v, tpr_v, color='lime', label='valid')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_title('ROC Curve')
    ax.legend(shadow=False, fontsize=14)
    return ax

def plot_everything(history, yt, yt_pred, yv, yv_pred):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    x = [i for i in range(len(history['loss_t']))]
    axes[0].plot(x, history['loss_t'], label='train loss', color='royalblue')
    axes[0].plot(x, history['loss_v'], label='valid loss', color='lime')
    
    axes[1].plot(x, history['auc_t'], label='train score', color='royalblue')
    axes[1].plot(x, history['auc_v'], label='valid score', color='lime')

    pr_t, rec_t, _ = precision_recall_curve(yt, yt_pred)
    pr_v, rec_v, _ = precision_recall_curve(yv, yv_pred)
    fpr_t, tpr_t, _ = roc_curve(yt, yt_pred)
    fpr_v, tpr_v, _ = roc_curve(yv, yv_pred)
    _ = draw_PR(rec_t, pr_t, rec_v, pr_v, ax=axes[2])
    _ = draw_ROC(fpr_t, tpr_t, fpr_v, tpr_v, ax=axes[3])
    
    for i, title in enumerate(['Loss', 'Auc Score']):
        axes[i].set_title(title)
        axes[i].set_ylabel(title)
        axes[i].set_xlabel('num_models')
        axes[i].legend(shadow=False, fontsize=14)
        
    plt.show()