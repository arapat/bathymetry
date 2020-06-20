import lightgbm as lgb
import pickle
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from .common import print_ts
from .load_data import persist_model


def train(config, train_features, train_labels, valid_features, valid_labels, logger):
    gbm_config = get_config(config)
    logger.log("booster, constrct dataset")
    train_dataset = lgb.Dataset(train_features, label=train_labels,
                                params={'max_bin': config["max_bin"]})
    valid_sets = [train_dataset]
    if valid_features is not None:
        valid_dataset = lgb.Dataset(valid_features, label=valid_labels,
                                    params={'max_bin': config["max_bin"]})
        valid_sets.append(valid_dataset)

    logger.log("booster, start training")
    try:
        gbm = lgb.train(
            gbm_config,
            train_dataset,
            num_boost_round=config["rounds"],
            valid_sets=valid_sets,
            callbacks=[print_ts(logger)],
        )
    except Exception as e:
        logger.log("training failed, {}".format(e))
        return None
    return gbm


def cv(config, data, nfolds, shuffle, logger):
    gbm_config = get_config(config)
    logger.log("booster, cv, reshape dataset")
    data = data.reshape((-1, data.shape[-1]))
    features, labels = data[:, 1:], data[:, 1].astype(np.int)
    logger.log("booster, cv, constrct dataset")
    train_dataset = lgb.Dataset(features, label=labels, params={'max_bin': config["max_bin"]})
    logger.log("booster, start cross validation")
    cv_results = lgb.cv(gbm_config, train_dataset, num_boost_round=config["rounds"],
                        nfold=nfolds, shuffle=shuffle, metrics=["binary_error", "binary", "auc"],
                        eval_train_metric=False, verbose_eval=True)
    return cv_results


def test(model, region, test_region, features, labels, logger):
    logger.log("booster, start predicting")
    preds = model.predict(features)
    scores = np.clip(preds, 1e-15, 1.0 - 1e-15)
    logger.log("booster, finish predicting")

    # compute auprc
    loss = np.mean(labels * -np.log(scores) + (1 - labels) * -np.log(1.0 - scores))
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    auprc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    auroc = auc(fpr, tpr)
    # accuracy
    acc = np.sum(labels == (scores > 0.5)) / labels.shape[0]

    logger.log("eval, {}, {}, {}, {}, {}, {}, {}".format(
        region, test_region, model.num_trees(), loss, auprc, auroc, acc))
    return scores


def get_config(config):
    return {
        "objective": config["objective"],
        "boosting_type": config["boosting_type"],
        "learning_rate": config["learning_rate"],
        "tree_learner": config["tree_learner"],
        "task": config["task"],
        "num_thread": config["num_thread"],
        "min_data_in_leaf": config["min_data_in_leaf"],
        "two_round": config["two_round"],
        "is_unbalance": config["is_unbalance"],
        "num_leaves": config["num_leaves"],
        "max_bin": config["max_bin"],
    }