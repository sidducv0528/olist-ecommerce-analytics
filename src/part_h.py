# -*- coding: utf-8 -*-
"""Part H: RFM + KMeans customer segmentation"""
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({'figure.dpi':100,'font.size':11,'axes.titlesize':13,'axes.labelsize':11})
sns.set_style('whitegrid')
RANDOM_STATE = 42
FIG_DIR = 'outputs/figures'
SAMPLE_SIZE = 20000   # subsample for silhouette (fast)

delivered = pd.read_csv('data/delivered.csv',
    parse_dates=['order_purchase_timestamp'])
print(f"[load] delivered: {delivered.shape}  ({time.time()-t0:.1f}s)")

# ── H. Customer Segmentation ────────────────────────────────────────────────
rfm_df = delivered.dropna(subset=['order_purchase_timestamp','total_price']).copy()
snap   = rfm_df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
print(f"  Snapshot date: {snap.date()}")

rfm = rfm_df.groupby('customer_id').agg(
    recency   = ('order_purchase_timestamp', lambda x: (snap - x.max()).days),
    frequency = ('order_id',   'nunique'),
    monetary  = ('total_price','sum')
).reset_index()
print(f"  RFM table: {rfm.shape}")
print(rfm[['recency','frequency','monetary']].describe().round(2).to_string())

# RFM quintile scoring
rfm['R_score'] = pd.qcut(rfm['recency'],   5, labels=[5,4,3,2,1]).astype(int)
rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'),  5, labels=[1,2,3,4,5]).astype(int)
rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
print(f"\n  RFM score range: {rfm['RFM_score'].min()} to {rfm['RFM_score'].max()}")
print(f"[H] RFM scored  ({time.time()-t0:.1f}s)")

# Scale for KMeans
scaler = StandardScaler()
X_all  = scaler.fit_transform(rfm[['recency','frequency','monetary']])

# Subsample for silhouette (speed)
rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(X_all), size=min(SAMPLE_SIZE, len(X_all)), replace=False)
X_samp = X_all[idx]

inertias, sil_scores = [], []
K_range = range(2, 9)
print(f"\n  Elbow+Silhouette (subsample={len(X_samp):,}) ...")
for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_samp)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_samp, km.labels_)
    sil_scores.append(sil)
    print(f"    k={k}  inertia={km.inertia_:>10,.1f}  silhouette={sil:.4f}  ({time.time()-t0:.1f}s)")

best_k = list(K_range)[int(np.argmax(sil_scores))]
print(f"\n  Best k = {best_k}  (silhouette={max(sil_scores):.4f})")

# Elbow + silhouette plot
fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].plot(list(K_range), inertias, 'bo-', ms=6)
axes[0].set_title('KMeans Elbow Curve')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
axes[0].set_xticks(list(K_range))
axes[1].plot(list(K_range), sil_scores, 'rs-', ms=6)
axes[1].axvline(best_k, color='green', ls='--', alpha=0.7, label=f'Best k={best_k}')
axes[1].set_title('Silhouette Score by k')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette Score')
axes[1].set_xticks(list(K_range)); axes[1].legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/12_kmeans_elbow_silhouette.png', bbox_inches='tight')
plt.close()
print(f"  Saved: 12_kmeans_elbow_silhouette.png  ({time.time()-t0:.1f}s)")

# Final fit on full data
km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
rfm['cluster'] = km_final.fit_predict(X_all)
print(f"  Final KMeans fit on {len(X_all):,} customers  ({time.time()-t0:.1f}s)")

profile = rfm.groupby('cluster').agg(
    count         = ('customer_id','count'),
    avg_recency   = ('recency',   'mean'),
    avg_frequency = ('frequency', 'mean'),
    avg_monetary  = ('monetary',  'mean'),
    avg_rfm_score = ('RFM_score', 'mean'),
).round(2)
print(f"\n  Cluster profiles:\n{profile.to_string()}")

# Name segments by descending RFM score
# Using descriptive names based on actual cluster profiles.
# Note: 96% of customers have frequency=1, so segmentation is driven by
# Recency and Monetary value only — frequency does not differentiate.
name_pool = ['High-Value Buyers','Recent Buyers','Lapsed Buyers',
             'Segment 4','Segment 5','Segment 6','Segment 7','Segment 8']
sorted_cl  = profile.sort_values('avg_rfm_score', ascending=False).index.tolist()
seg_map    = {cl: name_pool[i] if i < len(name_pool) else f'Segment {i+1}'
              for i, cl in enumerate(sorted_cl)}
rfm['segment'] = rfm['cluster'].map(seg_map)
profile['segment'] = [seg_map[i] for i in profile.index]

print(f"\n  Segment assignments:")
for cl, row in profile.sort_values('avg_rfm_score', ascending=False).iterrows():
    print(f"    Cluster {cl} -> {row['segment']:<24} "
          f"n={row['count']:,}  R={row['avg_recency']:.0f}d  "
          f"F={row['avg_frequency']:.2f}  M=R${row['avg_monetary']:.0f}  "
          f"RFM={row['avg_rfm_score']:.1f}")

# Scatter: Recency vs Monetary, bubble = count
colors_seg = ['#2ECC71','#3498DB','#9B59B6','#E74C3C',
              '#F39C12','#1ABC9C','#E67E22','#95A5A6']
fig, ax = plt.subplots(figsize=(11,7))
for i, (cl, row) in enumerate(profile.iterrows()):
    ax.scatter(row['avg_recency'], row['avg_monetary'],
               s=row['count']/8, alpha=0.75,
               color=colors_seg[i % len(colors_seg)],
               label=f"C{cl}: {row['segment']} (n={row['count']:,})")
ax.set_xlabel('Avg Recency (days — lower = more recent)')
ax.set_ylabel('Avg Monetary (R$)')
ax.set_title(f'RFM Clusters (k={best_k}) — bubble size proportional to customer count')
ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/13_rfm_clusters.png', bbox_inches='tight')
plt.close()
print(f"  Saved: 13_rfm_clusters.png")

# Segment bar chart
seg_dist = rfm['segment'].value_counts()
fig, ax  = plt.subplots(figsize=(10,5))
ax.bar(seg_dist.index, seg_dist.values, color='#4C9BE8', edgecolor='white')
ax.set_title('Customer Segment Distribution')
ax.set_xlabel('Segment'); ax.set_ylabel('Number of Customers')
plt.xticks(rotation=30, ha='right')
for i, v in enumerate(seg_dist.values):
    ax.text(i, v+10, f'{v:,}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/14_segment_distribution.png', bbox_inches='tight')
plt.close()
print(f"  Saved: 14_segment_distribution.png")

# Persist
rfm.to_csv('data/rfm_segments.csv', index=False)
with open('data/metrics_fg.json') as f:
    m = json.load(f)
m['best_k']          = int(best_k)
m['best_silhouette'] = float(max(sil_scores))
m['seg_map']         = {str(k): v for k,v in seg_map.items()}
m['cluster_sizes']   = {v: int((rfm['segment']==v).sum()) for v in seg_map.values()}
m['num_customers']   = int(len(rfm))
m['rfm_snapshot']    = str(snap.date())
with open('data/metrics_fg.json','w') as f:
    json.dump(m, f, indent=2)
print(f"  Updated: data/metrics_fg.json")

print(f"\n[H] Segmentation done  ({time.time()-t0:.1f}s)")
print("\n=== PART H COMPLETE ===")
