# -*- coding: utf-8 -*-
"""Part A-E: Imports, Load, Audit, Clean, Feature Engineering"""
import sys, time, warnings, os, json
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
t0 = time.time()

# ── A. Imports ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # must be before any other matplotlib import
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

plt.rcParams.update({
    'figure.dpi': 100, 'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 11, 'figure.figsize': (10, 5)
})
sns.set_style('whitegrid')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
FIG_DIR  = 'outputs/figures'
DATA_DIR = 'data'
os.makedirs(FIG_DIR,  exist_ok=True)
os.makedirs('models', exist_ok=True)
print(f"[A] Imports OK  ({time.time()-t0:.1f}s)")

# ── B. Data Loading ────────────────────────────────────────────────────────
orders      = pd.read_csv(f'{DATA_DIR}/olist_orders_dataset.csv')
items       = pd.read_csv(f'{DATA_DIR}/olist_order_items_dataset.csv')
payments    = pd.read_csv(f'{DATA_DIR}/olist_order_payments_dataset.csv')
reviews     = pd.read_csv(f'{DATA_DIR}/olist_order_reviews_dataset.csv')
customers   = pd.read_csv(f'{DATA_DIR}/olist_customers_dataset.csv')
products    = pd.read_csv(f'{DATA_DIR}/olist_products_dataset.csv')
sellers     = pd.read_csv(f'{DATA_DIR}/olist_sellers_dataset.csv')
translation = pd.read_csv(f'{DATA_DIR}/product_category_name_translation.csv')
# geolocation skipped – not used in analysis

tables = {
    'orders': orders, 'items': items, 'payments': payments,
    'reviews': reviews, 'customers': customers,
    'products': products, 'sellers': sellers, 'translation': translation
}

print(f"\n{'Table':<20} {'Rows':>8} {'Cols':>6}  {'Memory':>10}")
print("-" * 50)
for name, df in tables.items():
    mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  {name:<18} {len(df):>8,} {len(df.columns):>6}  {mem:>8.2f} MB")
print(f"[B] Data loaded  ({time.time()-t0:.1f}s)")

# ── C. Data Quality Audit ──────────────────────────────────────────────────
print("\n--- Missing values ---")
missing_summary = {}
for name, df in tables.items():
    miss     = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    has_miss = miss[miss > 0]
    if len(has_miss):
        missing_summary[name] = has_miss.to_dict()
        print(f"  {name}:")
        for col, cnt in has_miss.items():
            print(f"    {col}: {cnt:,}  ({miss_pct[col]:.1f}%)")
    else:
        print(f"  {name}: no missing values")

print("\n--- Duplicate rows ---")
for name, df in tables.items():
    n = df.duplicated().sum()
    print(f"  {name}: {n:,} duplicate rows")

dup_order  = orders['order_id'].duplicated().sum()
multi_rev  = reviews.groupby('order_id').size().gt(1).sum()
print(f"\n  Duplicate order_id in orders: {dup_order}")
print(f"  Orders with >1 review:        {multi_rev:,}")

# Missing-value bar chart
all_miss = {
    f"{name}.{col}": round(cnt / len(tables[name]) * 100, 1)
    for name, d in missing_summary.items()
    for col, cnt in d.items()
}
if all_miss:
    cols_sorted = sorted(all_miss, key=all_miss.get, reverse=True)
    fig, ax = plt.subplots(figsize=(12, max(4, len(cols_sorted) * 0.38)))
    ax.barh(cols_sorted, [all_miss[c] for c in cols_sorted], color='#e05c5c')
    ax.set_xlabel('Missing %')
    ax.set_title('Missing Values (%) Before Cleaning')
    ax.axvline(5, ls='--', color='gray', alpha=0.5, label='5% threshold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/00_missing_values.png', bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: 00_missing_values.png")

print(f"[C] Quality audit done  ({time.time()-t0:.1f}s)")

# ── D. Data Cleaning ──────────────────────────────────────────────────────
print("\n--- Cleaning ---")
cleaning_log = []

def log_action(table, action, before, after):
    removed = before - after
    cleaning_log.append({
        'table': table, 'action': action,
        'before': before, 'removed': removed, 'after': after
    })
    print(f"  [{table}] {action}: {before:,} -> {after:,}  (removed {removed:,})")

# 1. Drop duplicate rows
for name in ['orders','items','payments','reviews','customers','products','sellers']:
    df = tables[name]
    b  = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    log_action(name, 'drop_duplicates', b, len(df))
    tables[name] = df

orders=tables['orders']; items=tables['items']; payments=tables['payments']
reviews=tables['reviews']; customers=tables['customers']
products=tables['products']; sellers=tables['sellers']

# 2. Keep latest review per order (resolve 547 multi-review orders)
b = len(reviews)
reviews['review_creation_date'] = pd.to_datetime(
    reviews['review_creation_date'], errors='coerce')
reviews = (reviews
           .sort_values('review_creation_date', ascending=False)
           .drop_duplicates(subset='order_id', keep='first')
           .reset_index(drop=True))
log_action('reviews', 'keep_latest_review_per_order', b, len(reviews))

# 3. Convert date columns to datetime
date_map = {
    'orders':  ['order_purchase_timestamp','order_approved_at',
                'order_delivered_carrier_date','order_delivered_customer_date',
                'order_estimated_delivery_date'],
    'items':   ['shipping_limit_date'],
    'reviews': ['review_creation_date','review_answer_timestamp'],
}
for tbl_name, cols in date_map.items():
    for col in cols:
        if col in tables[tbl_name].columns:
            tables[tbl_name][col] = pd.to_datetime(
                tables[tbl_name][col], errors='coerce')
orders=tables['orders']; items=tables['items']; reviews=tables['reviews']
print("  Date columns -> datetime  OK")

# 4. Fill missing values
# products.product_category_name -> 'unknown'
products['product_category_name'] = (
    products['product_category_name'].fillna('unknown'))
print("  products.product_category_name: NaNs -> 'unknown'")

# products weight/dimensions -> median
for col in ['product_weight_g','product_length_cm',
            'product_height_cm','product_width_cm']:
    med = products[col].median()
    products[col] = products[col].fillna(med)
print("  products weight/dimension NaNs -> median")

# orders date NaNs: leave (rows excluded from delivery-only analysis)
# reviews comment columns: not used in ML, leave as-is

# 5. Translate product categories
products = products.merge(translation, on='product_category_name', how='left')
products['product_category_name_english'] = (
    products['product_category_name_english']
    .fillna(products['product_category_name']))
print(f"  Categories translated: "
      f"{products['product_category_name_english'].nunique()} unique EN categories")

# 6. Aggregate items per order  (avoids row multiplication on join)
items_agg = items.groupby('order_id').agg(
    total_price   = ('price',        'sum'),
    total_freight = ('freight_value','sum'),
    num_items     = ('order_item_id','count'),
    seller_id     = ('seller_id',    'first'),
    product_id    = ('product_id',   'first'),
).reset_index()
print(f"  items_agg: {len(items_agg):,} order-level rows")

# 7. Aggregate payments per order
pay_agg = payments.groupby('order_id').agg(
    payment_value        = ('payment_value',       'sum'),
    payment_type         = ('payment_type',
                            lambda x: x.mode().iloc[0] if len(x) else 'unknown'),
    payment_installments = ('payment_installments','max'),
).reset_index()
print(f"  pay_agg:   {len(pay_agg):,} order-level rows")

# 8. Build master dataframe
master = orders.merge(
    customers[['customer_id','customer_state']], on='customer_id', how='left')
master = master.merge(items_agg,   on='order_id', how='left')
master = master.merge(pay_agg,     on='order_id', how='left')
master = master.merge(
    reviews[['order_id','review_score']], on='order_id', how='left')
master = master.merge(
    products[['product_id','product_category_name_english']],
    on='product_id', how='left')
master = master.merge(
    sellers[['seller_id','seller_state']], on='seller_id', how='left')
print(f"  master shape: {master.shape}")

# 9. IQR capping for price and freight (1st–99th percentile)
def iqr_cap(series, name):
    q1  = series.quantile(0.01)
    q99 = series.quantile(0.99)
    print(f"  IQR cap [{name}]: [{q1:.2f}, {q99:.2f}]")
    return series.clip(lower=q1, upper=q99)

master['total_price']   = iqr_cap(master['total_price'].fillna(0),   'total_price')
master['total_freight'] = iqr_cap(master['total_freight'].fillna(0), 'total_freight')

# 10. Save
master.to_csv(f'{DATA_DIR}/cleaned_master.csv', index=False)
print(f"  Saved: data/cleaned_master.csv  ({master.shape[0]:,} x {master.shape[1]})")

# Save cleaning log
with open(f'{DATA_DIR}/cleaning_log.json','w') as f:
    json.dump(cleaning_log, f, indent=2)

print(f"[D] Cleaning done  ({time.time()-t0:.1f}s)")

# ── E. Feature Engineering ────────────────────────────────────────────────
print("\n--- Feature Engineering ---")
df = master.copy()

for col in ['order_purchase_timestamp',
            'order_delivered_customer_date',
            'order_estimated_delivery_date']:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df['delivery_days'] = (
    df['order_delivered_customer_date'] -
    df['order_purchase_timestamp']).dt.days

df['estimated_days'] = (
    df['order_estimated_delivery_date'] -
    df['order_purchase_timestamp']).dt.days

df['estimated_vs_actual_delay_days'] = (
    df['order_delivered_customer_date'] -
    df['order_estimated_delivery_date']).dt.days

df['is_late']        = (df['estimated_vs_actual_delay_days'] > 0).astype(int)
df['purchase_month'] = df['order_purchase_timestamp'].dt.month
df['purchase_weekday']= df['order_purchase_timestamp'].dt.dayofweek
df['purchase_hour']  = df['order_purchase_timestamp'].dt.hour
df['purchase_year']  = df['order_purchase_timestamp'].dt.year
df['freight_ratio']  = np.where(
    df['total_price'] > 0,
    df['total_freight'] / df['total_price'], 0.0)

# Fill remaining categorical NaNs
df['num_items']                     = df['num_items'].fillna(1)
df['payment_installments']          = df['payment_installments'].fillna(1)
df['payment_type']                  = df['payment_type'].fillna('unknown')
df['product_category_name_english'] = (
    df['product_category_name_english'].fillna('unknown'))
df['seller_state']   = df['seller_state'].fillna('unknown')
df['customer_state'] = df['customer_state'].fillna('unknown')

# Target variable (NO leakage: review_score is the source; low_review is derived)
df['low_review'] = (df['review_score'] <= 2).astype(int)

# Collapse rare product categories (< 100 orders) to 'other'
cat_counts = df['product_category_name_english'].value_counts()
df['product_category_name_english'] = df[
    'product_category_name_english'].replace(
    cat_counts[cat_counts < 100].index, 'other')

# delivered-only subset
delivered = df[df['order_status'] == 'delivered'].copy()

# Persist
df.to_csv(f'{DATA_DIR}/df_features.csv', index=False)
delivered.to_csv(f'{DATA_DIR}/delivered.csv', index=False)

del_data = delivered['delivery_days'].dropna()
print(f"  df shape:             {df.shape}")
print(f"  delivered orders:     {len(delivered):,}")
print(f"  delivery_days mean:   {del_data.mean():.1f}d")
print(f"  delivery_days median: {del_data.median():.1f}d")
print(f"  is_late:              {delivered['is_late'].mean()*100:.1f}% of delivered orders")
print(f"  low_review (score<=2):{delivered['low_review'].mean()*100:.1f}% of reviewed orders")
print(f"  product categories:   {df['product_category_name_english'].nunique()} unique (after collapsing rare)")

print(f"\n[E] Feature engineering done  ({time.time()-t0:.1f}s)")
print("\n=== PART A-E COMPLETE ===")
