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

def make_feature_list():
    features = '''00 lon
    01 lat
    02 depth
    03 pred
    04 (pred-depth)/depth
    05 d10
    06 d20
    07 d60
    08 age
    09 VGG
    10 rate
    11 sed
    12 roughness
    13 G:T
    14 NDP2.5m
    15 NDP5m
    16 NDP10m
    17 NDP30m
    18 STD2.5m
    19 STD5m
    20 STD10m
    21 STD30m
    22 MED2.5m
    23 MED5m
    24 MED10m
    25 MED30m
    26 D-MED2.5m/STD2.5m
    27 D-MED5m/STD5m
    28 D-MED10m/STD10m
    29 D-MED30m/STD30m
    30 year
    31 kind
    32 pred-abs(VGG_5m)'''.split('\n')
    features = [s.split()[1] for s in features]
    return features

def remove_feature_from_list(features,rm_feature):
    for x in rm_feature:
        try:
            features.remove(x)
        except:
            continue
    return features

def get_models(base_path):
    model_path = os.path.join(base_path, "runtime_models")
    models = [os.path.join(model_path, filename) for filename in os.listdir(model_path)
            if filename.endswith("pkl")]
    return models

def plot_ftr_importance(models,features):
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

def get_margin_plot(scores0, weights0, scores1, weights1, labels, ax, legends=None, title=None, colors=['b', 'r']):
    y0 = np.cumsum(weights0) / np.sum(weights0)
    ax[0].plot(scores0, 1.0 - y0, colors[0], label=labels[0])
    ax[0].fill_between(scores0, 1.0 - y0, alpha=0.2, color=colors[0])
    y1 = np.cumsum(weights1) / np.sum(weights1)
    ax[0].plot(scores1, y1, colors[1], label=labels[1])
    ax[0].fill_between(scores1, y1, alpha=0.2, color=colors[1])
    ax[0].legend(loc=9)
    ax[0].set_xlabel('Margin Score')
    ax[0].set_ylabel('Weights %')
    if title:
        ax[0].set_title(title)

    y0 = np.cumsum(np.ones_like(weights0)) / weights0.shape[0]
    ax[1].plot(scores0, 1.0 - y0, colors[0], label=labels[0])
    ax[1].fill_between(scores0, 1.0 - y0, alpha=0.2, color=colors[0])
    y1 = np.cumsum(np.ones_like(weights1)) / weights1.shape[0]
    ax[1].plot(scores1, y1, colors[1], label=labels[1])
    ax[1].fill_between(scores1, y1, alpha=0.2, color=colors[1])
    ax[1].legend(loc=9)
    ax[1].set_xlabel('Margin Score')
    ax[1].set_ylabel('# Measures %')
    if title:
        ax[1].set_title(title)
    return

def plot_scores(data, source):    
    _, labels, scores, weights = data
    scores0 = scores[labels == 0]
    weights0 = weights[labels == 0]
    order0 = np.argsort(scores0)
    scores0, weights0 = scores0[order0], weights0[order0]

    scores1 = scores[labels == 1]
    weights1 = weights[labels == 1]    
    order1 = np.argsort(scores1)
    scores1, weights1 = scores1[order1], weights1[order1]

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    # for i in range(10):
    get_margin_plot(scores0, weights0, scores1, weights1,
                    ["bad", "good"], ax)

    for i in range(len(ax)):
        ax[i].grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax[i].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

    ax[0].set_title('%s aggr. by weights\ndata_size=%d' % (source, labels.shape[0]))
    ax[1].set_title('%s aggr. by counts\ndata_size=%d' % (source, labels.shape[0]))
    return