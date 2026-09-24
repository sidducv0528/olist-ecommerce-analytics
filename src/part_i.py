# -*- coding: utf-8 -*-
"""Part I: Machine Learning – predict low_review (fast, no SHAP re-run)"""
import sys, time, warnings, os, json
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
t0 = time.time()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import xgboost as xgb

plt.rcParams.update({'figure.dpi':100,'font.size':11,'axes.titlesize':13,'axes.labelsize':11})
sns.set_style('whitegrid')
RANDOM_STATE = 42
FIG_DIR  = 'outputs/figures'
TUNE_N   = 40_000

def ts(label):
    print(f"  [{time.strftime('%H:%M:%S')}] {label}  ({time.time()-t0:.1f}s)")

# ── Load ─────────────────────────────────────────────────────────────────────
delivered = pd.read_csv('data/delivered.csv')
with open('data/metrics_fg.json') as f:
    metrics = json.load(f)
ts(f"Loaded delivered: {delivered.shape}")

# ── Feature / target definitions ─────────────────────────────────────────────
NUM_FEATS = ['delivery_days','estimated_vs_actual_delay_days','is_late',
             'total_price','total_freight','freight_ratio',
             'num_items','payment_installments',
             'purchase_month','purchase_weekday','purchase_hour']
CAT_FEATS = ['payment_type','product_category_name_english',
             'customer_state','seller_state']
ALL_FEATS = NUM_FEATS + CAT_FEATS
TARGET    = 'low_review'

ml_df = (delivered
         .dropna(subset=[TARGET,'delivery_days','estimated_vs_actual_delay_days'])
         .dropna(subset=ALL_FEATS)
         .copy())
ts(f"ML dataset: {ml_df.shape}  class balance: {ml_df[TARGET].value_counts().to_dict()}")

X = ml_df[ALL_FEATS]
y = ml_df[TARGET]

# ── Stratified 80/20 split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
ts(f"Train: {X_train.shape}  Test: {X_test.shape}  "
   f"Train-pos: {y_train.sum():,}/{len(y_train):,} ({y_train.mean()*100:.1f}%)")

# ── Preprocessing ─────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), NUM_FEATS),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_FEATS),
])
X_tr_pre = preprocessor.fit_transform(X_train)
X_te_pre = preprocessor.transform(X_test)
ts(f"Preprocessed shapes – train: {X_tr_pre.shape}  test: {X_te_pre.shape}")

# ── Tuning subsample (stratified 40k) ────────────────────────────────────────
rng  = np.random.default_rng(RANDOM_STATE)
idx0 = np.where(y_train.values == 0)[0]
idx1 = np.where(y_train.values == 1)[0]
n1   = int(TUNE_N * y_train.mean())
n0   = TUNE_N - n1
s0   = rng.choice(idx0, size=n0, replace=False)
s1   = rng.choice(idx1, size=n1, replace=False)
tune_idx = np.concatenate([s0, s1])
X_tune = X_tr_pre[tune_idx]
y_tune = y_train.values[tune_idx]
ts(f"Tuning subsample: {X_tune.shape}  pos={y_tune.sum():,}/{len(y_tune):,} ({y_tune.mean()*100:.1f}%)")

results = {}

def eval_model(name, clf, X_tr, y_tr, X_te, y_te):
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:,1]
    r = {
        'accuracy':  float(accuracy_score(y_te, y_pred)),
        'precision': float(precision_score(y_te, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_te, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_te, y_pred, zero_division=0)),
        'roc_auc':   float(roc_auc_score(y_te, y_prob)),
        'y_prob':    y_prob,
    }
    ts(f"{name:<22} Acc={r['accuracy']:.4f}  P={r['precision']:.4f}  "
       f"R={r['recall']:.4f}  F1={r['f1']:.4f}  AUC={r['roc_auc']:.4f}")
    return r

# ── Model 1: Logistic Regression ─────────────────────────────────────────────
print("\n  --- Logistic Regression ---")
lr = LogisticRegression(class_weight='balanced', max_iter=500,
                         solver='liblinear', random_state=RANDOM_STATE)
results['Logistic Regression'] = eval_model(
    'Logistic Regression', lr, X_tr_pre, y_train, X_te_pre, y_test)

# ── Model 2: Random Forest ────────────────────────────────────────────────────
print("\n  --- Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, max_depth=12,
                             class_weight='balanced',
                             random_state=RANDOM_STATE, n_jobs=-1)
results['Random Forest'] = eval_model(
    'Random Forest', rf, X_tr_pre, y_train, X_te_pre, y_test)

# ── Model 3: XGBoost ──────────────────────────────────────────────────────────
print("\n  --- XGBoost ---")
neg = int((y_train == 0).sum()); pos = int((y_train == 1).sum())
spw = round(neg / pos, 2)
xgb_clf = xgb.XGBClassifier(
    n_estimators=150, tree_method='hist', scale_pos_weight=spw,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    eval_metric='logloss')
results['XGBoost'] = eval_model(
    'XGBoost', xgb_clf, X_tr_pre, y_train, X_te_pre, y_test)

# ── Pick best by F1 ──────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['f1'])
print(f"\n  Best model by F1: {best_name}")

# ── Tune best model (n_iter=5, cv=3, subsample) ───────────────────────────────
print(f"\n  --- Tuning {best_name} (n_iter=5, cv=3) ---")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

if best_name == 'Random Forest':
    param_dist = {
        'n_estimators': [100, 150, 200],
        'max_depth':    [8, 12, 16, None],
        'min_samples_leaf': [1, 2, 4],
    }
    base = RandomForestClassifier(class_weight='balanced',
                                  random_state=RANDOM_STATE, n_jobs=-1)
elif best_name == 'XGBoost':
    param_dist = {
        'n_estimators':  [100, 150, 200],
        'max_depth':     [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample':     [0.8, 1.0],
    }
    base = xgb.XGBClassifier(tree_method='hist', scale_pos_weight=spw,
                               random_state=RANDOM_STATE, n_jobs=-1,
                               verbosity=0, eval_metric='logloss')
else:
    param_dist = {'C': [0.01, 0.1, 1, 10]}
    base = LogisticRegression(class_weight='balanced', max_iter=500,
                               solver='liblinear', random_state=RANDOM_STATE)

rscv = RandomizedSearchCV(base, param_dist, n_iter=5, cv=cv,
                           scoring='f1', random_state=RANDOM_STATE, n_jobs=-1,
                           verbose=1)
rscv.fit(X_tune, y_tune)
ts(f"Tuning done. Best params: {rscv.best_params_}  CV-F1: {rscv.best_score_:.4f}")

# Final fit on full training set
best_clf = rscv.best_estimator_
best_clf.fit(X_tr_pre, y_train)
y_pred_best = best_clf.predict(X_te_pre)
y_prob_best = best_clf.predict_proba(X_te_pre)[:,1]
tuned = {
    'accuracy':  float(accuracy_score(y_test, y_pred_best)),
    'precision': float(precision_score(y_test, y_pred_best, zero_division=0)),
    'recall':    float(recall_score(y_test, y_pred_best, zero_division=0)),
    'f1':        float(f1_score(y_test, y_pred_best, zero_division=0)),
    'roc_auc':   float(roc_auc_score(y_test, y_prob_best)),
}
ts(f"Tuned {best_name}: Acc={tuned['accuracy']:.4f}  P={tuned['precision']:.4f}  "
   f"R={tuned['recall']:.4f}  F1={tuned['f1']:.4f}  AUC={tuned['roc_auc']:.4f}")

# ── Model comparison table ────────────────────────────────────────────────────
print(f"\n  {'Model':<28} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print("  " + "-"*61)
for name, m in results.items():
    marker = " <-- pre-tune best" if name == best_name else ""
    print(f"  {name:<28} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
          f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['roc_auc']:>7.4f}{marker}")
label = f"Tuned {best_name}"
print(f"  {label:<28} {tuned['accuracy']:>7.4f} {tuned['precision']:>7.4f} "
      f"{tuned['recall']:>7.4f} {tuned['f1']:>7.4f} {tuned['roc_auc']:>7.4f}  <-- FINAL")

# ── Confusion matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()
print(f"\n  Confusion matrix:  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Not Low','Low Review'],
            yticklabels=['Not Low','Low Review'])
ax.set_title(f'Confusion Matrix - {best_name} (Tuned)')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/15_confusion_matrix.png', bbox_inches='tight')
plt.close()
ts("Saved: 15_confusion_matrix.png")

# ── ROC curves ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,6))
for name, m in results.items():
    fpr, tpr, _ = roc_curve(y_test, m['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={m['roc_auc']:.3f})")
fpr_t, tpr_t, _ = roc_curve(y_test, y_prob_best)
ax.plot(fpr_t, tpr_t, '--', lw=2,
        label=f"{best_name} Tuned (AUC={tuned['roc_auc']:.3f})")
ax.plot([0,1],[0,1],'k--', alpha=0.4, label='Random')
ax.set_title('ROC Curves - All Models')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/16_roc_curves.png', bbox_inches='tight')
plt.close()
ts("Saved: 16_roc_curves.png")

# ── Feature importance ────────────────────────────────────────────────────────
cat_enc_names  = preprocessor.named_transformers_['cat'].get_feature_names_out(CAT_FEATS)
all_feat_names = NUM_FEATS + list(cat_enc_names)
top_feat = 'N/A'

if hasattr(best_clf, 'feature_importances_'):
    imp = pd.Series(best_clf.feature_importances_, index=all_feat_names)
    imp.sort_values(ascending=False, inplace=True)
    top_feat = str(imp.index[0])
    fig, ax = plt.subplots(figsize=(10,7))
    imp.head(20)[::-1].plot.barh(ax=ax, color='#4C9BE8')
    ax.set_title(f'Top 20 Feature Importances - {best_name} (Tuned)')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/17_feature_importance.png', bbox_inches='tight')
    plt.close()
    ts(f"Saved: 17_feature_importance.png  Top: {top_feat} ({imp.iloc[0]:.4f})")
    print(f"\n  Top 10 features:")
    for feat, val in imp.head(10).items():
        print(f"    {feat:<45} {val:.4f}")

# ── SHAP: skip if PNG already exists (takes ~8 min) ──────────────────────────
shap_path = f'{FIG_DIR}/18_shap_summary.png'
if os.path.exists(shap_path):
    ts(f"SHAP already exists ({os.path.getsize(shap_path)//1024} KB) - skipping re-run")
else:
    try:
        import shap
        ts("Computing SHAP on 300 samples (this takes several minutes)...")
        explainer = shap.TreeExplainer(best_clf)
        sv = explainer.shap_values(X_te_pre[:300])
        if isinstance(sv, list):
            sv = sv[1]
        fig, ax = plt.subplots(figsize=(10,7))
        shap.summary_plot(sv, X_te_pre[:300], feature_names=all_feat_names,
                          show=False, plot_type='bar')
        plt.title(f'SHAP Feature Importance - {best_name}')
        plt.tight_layout()
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()
        ts("Saved: 18_shap_summary.png")
    except Exception as e:
        print(f"  SHAP skipped: {e}")

# ── Save model ────────────────────────────────────────────────────────────────
payload = {
    'preprocessor': preprocessor,
    'classifier':   best_clf,
    'model_name':   best_name,
    'features':     ALL_FEATS,
    'num_features': NUM_FEATS,
    'cat_features': CAT_FEATS,
    'metrics':      tuned,
    'best_params':  {str(k): str(v) for k,v in rscv.best_params_.items()},
}
joblib.dump(payload, 'models/best_model.joblib')
ts("Saved: models/best_model.joblib")

# ── Update metrics JSON ────────────────────────────────────────────────────────
metrics['best_model_name']  = best_name
metrics['best_params']      = {str(k): str(v) for k,v in rscv.best_params_.items()}
metrics['tuned_metrics']    = tuned
metrics['model_comparison'] = {
    k: {mk: mv for mk,mv in v.items() if mk != 'y_prob'}
    for k,v in results.items()
}
metrics['confusion_matrix'] = cm.tolist()
metrics['top_feature']      = top_feat
metrics['scale_pos_weight'] = float(spw)
metrics['ml_train_size']    = int(len(X_train))
metrics['ml_test_size']     = int(len(X_test))
metrics['tune_sample_size'] = int(len(X_tune))
metrics['total_revenue']    = float(delivered['total_price'].sum())
metrics['total_orders']     = int(len(delivered))

with open('data/metrics_fg.json','w') as f:
    json.dump(metrics, f, indent=2)
ts("Updated: data/metrics_fg.json")

print(f"\n[I] Machine learning done  ({time.time()-t0:.1f}s)")
print("\n=== PART I COMPLETE ===")
print("\n=== ALL 4 PARTS DONE ===")
