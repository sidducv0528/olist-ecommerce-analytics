# -*- coding: utf-8 -*-
"""Part F-G: EDA visualisations + Statistical tests"""
import sys, time, warnings, os, json
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
t0 = time.time()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency, shapiro

plt.rcParams.update({'figure.dpi':100,'font.size':11,'axes.titlesize':13,
                     'axes.labelsize':11,'figure.figsize':(10,5)})
sns.set_style('whitegrid')
RANDOM_STATE = 42
FIG_DIR = 'outputs/figures'

delivered = pd.read_csv('data/delivered.csv',
    parse_dates=['order_purchase_timestamp',
                 'order_delivered_customer_date',
                 'order_estimated_delivery_date'])
del_data = delivered['delivery_days'].dropna()
print(f"[load] delivered: {delivered.shape}  ({time.time()-t0:.1f}s)")

# ── F. EDA ─────────────────────────────────────────────────────────────────

# F1: Monthly revenue & orders
delivered['year_month'] = delivered['order_purchase_timestamp'].dt.to_period('M')
monthly = (delivered.groupby('year_month')
           .agg(revenue=('total_price','sum'), orders=('order_id','count'))
           .reset_index())
monthly['ym_str'] = monthly['year_month'].astype(str)

fig, ax1 = plt.subplots(figsize=(14,5))
ax2 = ax1.twinx()
ax1.bar(range(len(monthly)), monthly['revenue'],  color='#4C9BE8', alpha=0.7, label='Revenue (R$)')
ax2.plot(range(len(monthly)), monthly['orders'],   color='#E84C4C', marker='o', ms=4, label='Orders')
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(monthly['ym_str'], rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('Revenue (R$)')
ax2.set_ylabel('Number of Orders')
ax1.set_title('Monthly Revenue and Order Volume (Delivered Orders)')
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'R${x/1e6:.1f}M'))
h1,l1 = ax1.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper left')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/01_monthly_revenue_orders.png', bbox_inches='tight')
plt.close()
peak_month = monthly.loc[monthly['revenue'].idxmax(),'ym_str']
peak_rev   = monthly['revenue'].max()
print(f"[F1] Monthly trend saved.  Peak month: {peak_month}  R${peak_rev:,.0f}  ({time.time()-t0:.1f}s)")

# F2: Top 10 categories by revenue
top10_cat = (delivered.groupby('product_category_name_english')['total_price']
             .sum().sort_values(ascending=False).head(10).reset_index())
top10_cat.columns = ['category','revenue']
fig, ax = plt.subplots(figsize=(10,6))
bars = ax.barh(top10_cat['category'][::-1], top10_cat['revenue'][::-1], color='#4C9BE8')
ax.set_xlabel('Total Revenue (R$)')
ax.set_title('Top 10 Product Categories by Revenue')
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'R${x/1e6:.1f}M'))
for bar, val in zip(bars, top10_cat['revenue'][::-1]):
    ax.text(bar.get_width()+2000, bar.get_y()+bar.get_height()/2,
            f'R${val/1e6:.2f}M', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_top10_categories_revenue.png', bbox_inches='tight')
plt.close()
print(f"[F2] Top categories saved. #1: {top10_cat.iloc[0]['category']}  R${top10_cat.iloc[0]['revenue']:,.0f}  ({time.time()-t0:.1f}s)")

# F3: Orders by weekday & hour
wmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
delivered['weekday_name'] = delivered['purchase_weekday'].map(wmap)
wd = (delivered.groupby('weekday_name')['order_id'].count()
      .reindex(['Mon','Tue','Wed','Thu','Fri','Sat','Sun']))
hr = delivered.groupby('purchase_hour')['order_id'].count()
fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].bar(wd.index, wd.values, color='#7EC8A4')
axes[0].set_title('Orders by Day of Week')
axes[0].set_xlabel('Day'); axes[0].set_ylabel('Orders')
axes[1].plot(hr.index, hr.values, color='#E84C4C', marker='o', ms=4)
axes[1].fill_between(hr.index, hr.values, alpha=0.2, color='#E84C4C')
axes[1].set_title('Orders by Hour of Day')
axes[1].set_xlabel('Hour (0-23)'); axes[1].set_ylabel('Orders')
axes[1].set_xticks(range(0,24,2))
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/03_orders_weekday_hour.png', bbox_inches='tight')
plt.close()
peak_day = str(wd.idxmax())
peak_hr  = int(hr.idxmax())
print(f"[F3] Weekday/hour saved.  Busiest: {peak_day}, Peak hour: {peak_hr}:00  ({time.time()-t0:.1f}s)")

# F4: Payment type distribution
pay_dist = delivered['payment_type'].value_counts()
fig, ax = plt.subplots(figsize=(8,5))
pal = ['#4C9BE8','#E84C4C','#7EC8A4','#F5A623','#9B59B6']
ax.bar(pay_dist.index, pay_dist.values, color=pal[:len(pay_dist)])
ax.set_title('Payment Type Distribution')
ax.set_xlabel('Payment Type'); ax.set_ylabel('Orders')
for i,(k,v) in enumerate(pay_dist.items()):
    ax.text(i, v+50, f'{v:,}\n({v/len(delivered)*100:.1f}%)', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/04_payment_type.png', bbox_inches='tight')
plt.close()
dom_pay     = str(pay_dist.index[0])
dom_pay_pct = float(pay_dist.iloc[0]/len(delivered)*100)
print(f"[F4] Payment type saved.  Dominant: {dom_pay} ({dom_pay_pct:.1f}%)  ({time.time()-t0:.1f}s)")

# F5: Review score distribution
rev_dist = delivered['review_score'].dropna().astype(int).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8,5))
pal5 = ['#E84C4C','#F5A623','#F5E623','#7EC8A4','#4C9BE8']
ax.bar(rev_dist.index.astype(str), rev_dist.values, color=pal5)
ax.set_title('Review Score Distribution')
ax.set_xlabel('Review Score (1=worst, 5=best)'); ax.set_ylabel('Count')
for i,(k,v) in enumerate(rev_dist.items()):
    ax.text(i, v+50, f'{v:,}\n({v/rev_dist.sum()*100:.1f}%)', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/05_review_score_dist.png', bbox_inches='tight')
plt.close()
avg_review     = float(delivered['review_score'].mean())
low_review_pct = float((delivered['review_score']<=2).mean()*100)
print(f"[F5] Review dist saved.  Mean={avg_review:.2f}, Low-review={low_review_pct:.1f}%  ({time.time()-t0:.1f}s)")

# F6: Delivery time distribution
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(del_data, bins=50, color='#4C9BE8', edgecolor='white', alpha=0.8)
ax.axvline(del_data.mean(),   color='red',    ls='--', label=f'Mean={del_data.mean():.1f}d')
ax.axvline(del_data.median(), color='orange', ls='--', label=f'Median={del_data.median():.1f}d')
ax.set_title('Delivery Time Distribution'); ax.set_xlabel('Days'); ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/06_delivery_time_dist.png', bbox_inches='tight')
plt.close()
print(f"[F6] Delivery dist saved. Mean={del_data.mean():.1f}d, Median={del_data.median():.1f}d, Max={del_data.max():.0f}d  ({time.time()-t0:.1f}s)")

# F7: Late vs on-time pie
late_counts = delivered['is_late'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6,6))
ax.pie(late_counts.values, labels=['On-Time','Late'],
       autopct='%1.1f%%', colors=['#7EC8A4','#E84C4C'],
       startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
ax.set_title('Late vs On-Time Deliveries')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/07_late_vs_ontime.png', bbox_inches='tight')
plt.close()
late_pct = float(delivered['is_late'].mean()*100)
print(f"[F7] Late pie saved.  Late={late_pct:.1f}%  ({time.time()-t0:.1f}s)")

# F8: Avg delivery time by state
state_del = (delivered.groupby('customer_state')['delivery_days']
             .mean().sort_values(ascending=False).reset_index())
state_del.columns = ['state','avg_days']
fig, ax = plt.subplots(figsize=(14,6))
ax.bar(state_del['state'], state_del['avg_days'],
       color=['#E84C4C' if d>del_data.mean() else '#4C9BE8' for d in state_del['avg_days']])
ax.axhline(del_data.mean(), color='black', ls='--', alpha=0.5,
           label=f'Mean={del_data.mean():.1f}d')
ax.set_title('Average Delivery Time by Customer State')
ax.set_xlabel('State'); ax.set_ylabel('Avg Delivery Days')
ax.legend(); plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/08_delivery_by_state.png', bbox_inches='tight')
plt.close()
worst_state = str(state_del.iloc[0]['state']); worst_days = float(state_del.iloc[0]['avg_days'])
best_state  = str(state_del.iloc[-1]['state']); best_days  = float(state_del.iloc[-1]['avg_days'])
print(f"[F8] Delivery by state saved. Slowest: {worst_state}({worst_days:.1f}d)  Fastest: {best_state}({best_days:.1f}d)  ({time.time()-t0:.1f}s)")

# F9: Revenue by state
state_rev = (delivered.groupby('customer_state')['total_price']
             .sum().sort_values(ascending=False).reset_index())
state_rev.columns = ['state','revenue']
fig, ax = plt.subplots(figsize=(14,6))
ax.bar(state_rev['state'], state_rev['revenue'], color='#4C9BE8')
ax.set_title('Total Revenue by Customer State')
ax.set_xlabel('State'); ax.set_ylabel('Revenue (R$)')
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'R${x/1e6:.1f}M'))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/09_revenue_by_state.png', bbox_inches='tight')
plt.close()
top_rev_state = str(state_rev.iloc[0]['state'])
top_rev_val   = float(state_rev.iloc[0]['revenue'])
print(f"[F9] Revenue by state saved. Top: {top_rev_state}  R${top_rev_val:,.0f}  ({time.time()-t0:.1f}s)")

# F10: Freight ratio vs review score
fr_data = delivered.dropna(subset=['review_score','freight_ratio']).copy()
fr_data['rs'] = fr_data['review_score'].astype(int)
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(data=fr_data, x='rs', y='freight_ratio',
            palette=['#E84C4C','#F5A623','#F5E623','#7EC8A4','#4C9BE8'], ax=ax)
ax.set_title('Freight Ratio by Review Score')
ax.set_xlabel('Review Score'); ax.set_ylabel('Freight / Price Ratio')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/10_freight_ratio_review.png', bbox_inches='tight')
plt.close()
med_fr_low  = float(fr_data[fr_data['rs']<=2]['freight_ratio'].median())
med_fr_high = float(fr_data[fr_data['rs']>=4]['freight_ratio'].median())
print(f"[F10] Freight ratio saved. Low-review median={med_fr_low:.3f}, High-review={med_fr_high:.3f}  ({time.time()-t0:.1f}s)")

# F11: Correlation heatmap
num_cols = ['delivery_days','estimated_vs_actual_delay_days','total_price',
            'total_freight','freight_ratio','num_items','payment_installments',
            'review_score','is_late']
corr_mat = delivered[num_cols].dropna().corr()
fig, ax = plt.subplots(figsize=(10,8))
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, ax=ax, cbar_kws={'shrink':0.8})
ax.set_title('Correlation Heatmap (Delivered Orders)')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/11_correlation_heatmap.png', bbox_inches='tight')
plt.close()
corr_late_rev = float(corr_mat.loc['is_late','review_score'])
print(f"[F11] Heatmap saved. Corr(is_late, review_score)={corr_late_rev:.4f}  ({time.time()-t0:.1f}s)")
print(f"[F] EDA complete  ({time.time()-t0:.1f}s)")

# ── G. Statistical Tests ────────────────────────────────────────────────────
print("\n--- G. Statistical Tests ---")

late_scores   = delivered.loc[delivered['is_late']==1,'review_score'].dropna()
ontime_scores = delivered.loc[delivered['is_late']==0,'review_score'].dropna()
print(f"  Late   n={len(late_scores):,}  mean={late_scores.mean():.4f}  median={late_scores.median():.1f}")
print(f"  OnTime n={len(ontime_scores):,}  mean={ontime_scores.mean():.4f}  median={ontime_scores.median():.1f}")

# Normality (Shapiro-Wilk on 5000-sample subsample)
ss = 5000
_,p_nl = shapiro(late_scores.sample(min(ss,len(late_scores)),   random_state=RANDOM_STATE))
_,p_no = shapiro(ontime_scores.sample(min(ss,len(ontime_scores)),random_state=RANDOM_STATE))
print(f"\n  Shapiro-Wilk p (late   n={min(ss,len(late_scores)):,}):   {p_nl:.4e}")
print(f"  Shapiro-Wilk p (ontime n={min(ss,len(ontime_scores)):,}): {p_no:.4e}")
print(f"  Both p<<0.05 => distributions non-normal => Mann-Whitney U")

# Test 1: Mann-Whitney U
mw_stat, mw_p = mannwhitneyu(late_scores, ontime_scores, alternative='less')
n1, n2 = len(late_scores), len(ontime_scores)
effect_r = 1 - (2*mw_stat)/(n1*n2)
print(f"\n  H0: Scores equal for late vs on-time deliveries")
print(f"  H1: Late deliveries have LOWER review scores")
print(f"  U = {mw_stat:,.0f}   p = {mw_p:.4e}")
print(f"  Effect size r = {effect_r:.4f}  (|r|>=0.3 = large)")
if mw_p < 0.05:
    print(f"  => REJECT H0. Late deliveries have significantly lower scores.")
else:
    print(f"  => Fail to reject H0 at alpha=0.05.")

# Test 2: Chi-square + Cramer's V
ct = pd.crosstab(delivered['is_late'], delivered['low_review'])
print(f"\n  Contingency table (is_late x low_review):\n{ct.to_string()}")
chi2, p_chi, dof, _ = chi2_contingency(ct)
n_chi  = int(ct.values.sum())
cramers_v = float(np.sqrt(chi2 / (n_chi * (min(ct.shape)-1))))
print(f"\n  Chi2 = {chi2:.4f}   df = {dof}   p = {p_chi:.4e}")
print(f"  Cramer's V = {cramers_v:.4f}")
if p_chi < 0.05:
    print(f"  => Significant association between late delivery and low review (V={cramers_v:.4f}).")
else:
    print(f"  => No significant association at alpha=0.05.")

print(f"\n[G] Stats done  ({time.time()-t0:.1f}s)")

# ── Persist metrics ─────────────────────────────────────────────────────────
metrics = {
    'avg_review_score':      avg_review,
    'low_review_pct':        low_review_pct,
    'late_pct':              late_pct,
    'avg_delivery_days':     float(del_data.mean()),
    'median_delivery_days':  float(del_data.median()),
    'peak_month':            peak_month,
    'peak_day':              peak_day,
    'peak_hour':             peak_hr,
    'top_category':          str(top10_cat.iloc[0]['category']),
    'top_category_rev':      float(top10_cat.iloc[0]['revenue']),
    'worst_state':           worst_state,
    'worst_days':            worst_days,
    'best_state':            best_state,
    'best_days':             best_days,
    'top_rev_state':         top_rev_state,
    'top_rev_state_rev':     top_rev_val,
    'late_mean_score':       float(late_scores.mean()),
    'ontime_mean_score':     float(ontime_scores.mean()),
    'mannwhitney_u':         float(mw_stat),
    'mannwhitney_p':         float(mw_p),
    'effect_r':              float(effect_r),
    'chi2':                  float(chi2),
    'chi2_p':                float(p_chi),
    'cramers_v':             cramers_v,
    'corr_late_review':      corr_late_rev,
    'med_fr_low':            med_fr_low,
    'med_fr_high':           med_fr_high,
    'dominant_payment':      dom_pay,
    'dominant_payment_pct':  dom_pay_pct,
}
with open('data/metrics_fg.json','w') as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved: data/metrics_fg.json")
print("\n=== PART F-G COMPLETE ===")
