import os
import numpy as np
import pickle as pkl
import lightgbm as lgb
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import seaborn as sns
import pandas as pd


def get_models(base_path):
    model_path = os.path.join(base_path, "runtime_models")
    models = [os.path.join(model_path, filename) for filename in os.listdir(model_path)
            if filename.endswith("pkl")]
    return models

def plot_ftr_importance(models):
    all_feature_imp = []
    fig, ax = plt.subplots(len(models), 1, figsize=(10, 6 * len(models)))

    for ax, model_path in zip(ax, models):
        model_name = os.path.basename(model_path).split(".")[0]
        with open(model_path, 'rb') as f:
            model = pkl.load(f)
        imp = sorted(zip(model.feature_importance(importance_type='gain'),features))
        feature_imp = pd.DataFrame(imp, columns=['Value','Feature']) \
                        .sort_values(by="Value", ascending=False)
        feature_imp["agency"] = model_name.rsplit('_', 1)[0]
        all_feature_imp.append(feature_imp)

        sns.barplot(x="Value", y="Feature", data=feature_imp, ax=ax)
        ax.set_title('{} LightGBM Features'.format(model_name))

    fig.tight_layout()
    return


def get_performance(true, scores, weights):
    # loss
    loss = np.mean(true * -np.log(scores) + (1 - true) * -np.log(1.0 - scores))
    # auprc
    precision, recall, thr = precision_recall_curve(true, scores, pos_label=1)
    auprc = auc(recall, precision)
    # auroc
    fpr, tpr, _ = roc_curve(true, scores, pos_label=1)
    auroc = auc(fpr, tpr)
    # accuracy
    acc = np.sum(true == (scores > 0.5)) / true.shape[0]
    return (loss, acc, (precision, recall, thr, auprc), (fpr, tpr, auroc))

def plot_prc(ax,prec,rec,thresh,source,auprc):
    ax.plot(rec,prec)
    ax.grid()
    
    scatter_x, scatter_y = [], []
    last_recall = 1.0
    for x, y, thr in zip(rec, prec, thresh):
        if last_recall - x >= 0.1 or last_recall - x >= 0.05 and x >= 0.9:
            ax.annotate(">{:.2f}".format(thr),
                         (x,y),
                         textcoords="offset points",
                         xytext=(0,10),
                         ha='center')
            scatter_x.append(x)
            scatter_y.append(y)
            last_recall = x
    
    ax.scatter(scatter_x, scatter_y)
    ax.set_title("{} AUPRC={:.4f}".format(source, auprc));
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    return

def plot_roc(ax,fpr,tpr,source,auroc):
    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(fpr, tpr)
    ax.grid()
    ax.set_title("{} AUROC={:.4f}".format(source, auroc));
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    return

def plot_PRCROC(sources,base_dir):
    scores_filename = "model_{}_test_{}_scores.pkl"
    fig, ax_list = plt.subplots(1, 2, figsize=(20,8))
    for i, source in enumerate(sources):
        score_file = base_dir + scores_filename.format(source,source)
        with open(score_file,"rb") as f:
            (features, label, scores, weights) = pkl.load(f)
            
        (loss, acc, (precision, recall, thresh, auprc), _) = get_performance(label,scores,weights)
        axis = ax_list[i]
        plot_prc(axis,precision,recall,thresh,source,auprc)

    fig, ax_list = plt.subplots(1, 2, figsize=(20,8))
    for i, source in enumerate(sources):
        score_file = base_dir + scores_filename.format(source,source)
        with open(score_file,"rb") as f:
            (features, label, scores, weights) = pkl.load(f)
            
        (_, acc, _, (fpr, tpr, auroc)) = get_performance(label,scores,weights)
        axis = ax_list[i]
        plot_roc(axis,fpr,tpr,source,auroc)
    return