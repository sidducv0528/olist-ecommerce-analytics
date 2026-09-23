# -*- coding: utf-8 -*-
"""
build_report.py
Generates Siddu_Varikuppala_ProjectReport.docx from real metrics + saved figures.
Run after all part_*.py scripts have completed.
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

REPORT_PATH = 'Siddu_Varikuppala_ProjectReport.docx'
FIG_DIR     = 'outputs/figures'

# ── Load metrics ──────────────────────────────────────────────────────────────
with open('data/metrics_fg.json') as f:
    m = json.load(f)

# ── Helpers ───────────────────────────────────────────────────────────────────
doc = Document()

def set_margins(doc, top=1.0, bottom=1.0, left=1.2, right=1.2):
    from docx.shared import Inches
    for section in doc.sections:
        section.top_margin    = Inches(top)
        section.bottom_margin = Inches(bottom)
        section.left_margin   = Inches(left)
        section.right_margin  = Inches(right)

def add_title(text, level=0):
    if level == 0:
        p = doc.add_heading(text, level=0)
        p.runs[0].font.size = Pt(18)
        p.runs[0].font.color.rgb = RGBColor(0x1f, 0x23, 0x28)
    else:
        doc.add_heading(text, level=level)

def add_para(text, bold=False, italic=False, font_size=11, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(font_size)
    if align:
        p.alignment = align
    return p

def add_bullet(text):
    p = doc.add_paragraph(text, style='List Bullet')
    p.runs[0].font.size = Pt(11)
    return p

def add_figure(filename, caption, width=5.5):
    path = os.path.join(FIG_DIR, filename)
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        p = doc.add_paragraph(caption)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.size  = Pt(10)
        p.runs[0].font.italic = True
        p.runs[0].font.color.rgb = RGBColor(0x57, 0x60, 0x6a)
    else:
        doc.add_paragraph(f"[Figure not found: {path}]")

def add_table_row(table, cells, bold=False):
    row = table.add_row()
    for i, val in enumerate(cells):
        cell = row.cells[i]
        cell.text = str(val)
        if bold:
            for run in cell.paragraphs[0].runs:
                run.bold = True
    return row

def add_page_break():
    doc.add_page_break()

set_margins(doc)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE PAGE
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Olist Brazilian E-Commerce\nCustomer, Delivery & Review Analytics\nwith Machine Learning")
run.bold = True; run.font.size = Pt(20)
run.font.color.rgb = RGBColor(0x1f, 0x23, 0x28)

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Siddu Varikuppala")
run.bold = True; run.font.size = Pt(14)

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026\n"
          "BharatCares").font.size = Pt(12)

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run(f"Date: {datetime.date.today().strftime('%B %d, %Y')}").font.size = Pt(11)

add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# TABLE OF CONTENTS (manual)
# ══════════════════════════════════════════════════════════════════════════════
add_title("Table of Contents", level=1)
toc_items = [
    "Abstract", "1. Introduction", "2. Problem Statement & Objectives",
    "3. Dataset Description", "4. Tools & Technologies", "5. Methodology",
    "6. Data Cleaning", "7. Exploratory Data Analysis",
    "8. Statistical Analysis", "9. Customer Segmentation",
    "10. Machine Learning Model", "11. Streamlit App",
    "12. Business Recommendations", "13. Limitations & Future Scope",
    "14. Conclusion", "15. References",
    "16. Use of IBM Bob AI Tool",
]
for item in toc_items:
    add_bullet(item)
add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# ABSTRACT
# ══════════════════════════════════════════════════════════════════════════════
add_title("Abstract", level=1)
add_para(
    f"This project analyses the Brazilian E-Commerce Public Dataset by Olist "
    f"({m['total_orders']:,} delivered orders, 2016-2018) to understand the drivers of customer "
    f"review quality and to build a machine-learning model that predicts low-review orders "
    f"(score <= 2) without data leakage. Key findings: {m['late_pct']:.1f}% of orders are "
    f"delivered late, and late deliveries score a mean of {m['late_mean_score']:.2f} vs "
    f"{m['ontime_mean_score']:.2f} for on-time orders (Mann-Whitney U p<0.001, effect r={m['effect_r']:.2f}). "
    f"A tuned {m['best_model_name']} achieves ROC-AUC={m['tuned_metrics']['roc_auc']:.4f} and "
    f"F1={m['tuned_metrics']['f1']:.4f}. Five evidence-based business recommendations are presented."
)
add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ══════════════════════════════════════════════════════════════════════════════
add_title("1. Introduction", level=1)
add_para(
    "Olist is a Brazilian marketplace that connects small businesses to major e-commerce channels "
    "including Mercado Livre and B2W. The platform processes tens of thousands of orders monthly "
    "across all 27 Brazilian states. Customer reviews are a critical feedback signal: low scores "
    "damage seller reputation, reduce platform trust, and increase churn. This project applies "
    "end-to-end data analytics and machine learning to the publicly available Olist dataset to "
    "surface the root causes of poor reviews and build a predictive tool that operations teams "
    "can use to intervene before a bad review is posted."
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. PROBLEM STATEMENT
# ══════════════════════════════════════════════════════════════════════════════
add_title("2. Problem Statement & Objectives", level=1)
add_para("Problem Statement", bold=True)
add_para(
    "What factors drive low customer review scores (1-2) on the Olist platform, "
    "and can we predict whether a given order will receive a bad review before the "
    "customer posts it — without using any review data as a feature?"
)
add_para("Objectives", bold=True)
for obj in [
    "Explore sales, delivery and review trends across 2016-2018.",
    "Statistically test whether late delivery causes lower review scores.",
    "Segment customers using RFM scoring and KMeans clustering.",
    "Build an ML model to predict low-review orders (score <= 2) with no data leakage.",
    "Translate findings into five actionable business recommendations.",
]:
    add_bullet(obj)

# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASET DESCRIPTION
# ══════════════════════════════════════════════════════════════════════════════
add_title("3. Dataset Description", level=1)
add_para(
    "Source: Brazilian E-Commerce Public Dataset by Olist "
    "(https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). "
    "The dataset covers orders placed between September 2016 and August 2018."
)

tbl = doc.add_table(rows=1, cols=4)
tbl.style = 'Table Grid'
hdr = tbl.rows[0].cells
for i, h in enumerate(['File', 'Rows', 'Columns', 'Key Columns']):
    hdr[i].text = h
    hdr[i].paragraphs[0].runs[0].bold = True

dataset_info = [
    ('olist_orders_dataset.csv',        '99,441', '8',  'order_id, customer_id, status, timestamps'),
    ('olist_order_items_dataset.csv',   '112,650','7',  'order_id, product_id, seller_id, price, freight'),
    ('olist_order_payments_dataset.csv','103,886','5',  'order_id, payment_type, payment_value'),
    ('olist_order_reviews_dataset.csv', '99,224', '7',  'order_id, review_score, creation_date'),
    ('olist_customers_dataset.csv',     '99,441', '5',  'customer_id, customer_state, zip_code'),
    ('olist_products_dataset.csv',      '32,951', '9',  'product_id, category_name, weight, dims'),
    ('olist_sellers_dataset.csv',       '3,095',  '4',  'seller_id, seller_state, zip_code'),
    ('olist_geolocation_dataset.csv',   '1,000,163','5','zip_code, lat, lng (skipped in analysis)'),
    ('product_category_name_translation.csv','71','2',  'category_name -> English'),
]
for row in dataset_info:
    add_table_row(tbl, row)
doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# 4. TOOLS & TECHNOLOGIES
# ══════════════════════════════════════════════════════════════════════════════
add_title("4. Tools & Technologies", level=1)

tools_tbl = doc.add_table(rows=1, cols=3)
tools_tbl.style = 'Table Grid'
for i,h in enumerate(['Category','Tool / Library','Version']):
    tools_tbl.rows[0].cells[i].text = h
    tools_tbl.rows[0].cells[i].paragraphs[0].runs[0].bold = True

tools = [
    ('Language',    'Python',          '3.11.9'),
    ('Data',        'pandas',          '3.0.5'),
    ('Numerics',    'numpy',           '2.4.6'),
    ('Visualisation','matplotlib',     '3.11.2'),
    ('Visualisation','seaborn',        '0.13.2'),
    ('Visualisation','plotly',         '6.9.0'),
    ('ML',          'scikit-learn',    '1.9.0'),
    ('ML',          'xgboost',         '3.2.0'),
    ('ML',          'imbalanced-learn','0.14.2'),
    ('ML',          'shap',            '0.51.0'),
    ('Statistics',  'scipy',           '1.17.1'),
    ('App',         'streamlit',       '1.60.0'),
    ('Notebook',    'jupyter/nbformat','5.11.1'),
    ('Report',      'python-docx',     '1.1.2'),
    ('IDE',         'IBM Bob (AI)',     'N/A'),
]
for t in tools:
    add_table_row(tools_tbl, t)
doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# 5. METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
add_title("5. Methodology", level=1)
add_para(
    "The project follows a standard data science workflow:",
    bold=True
)
for step in [
    "Data Loading & Audit: load all 8 tables, check missing values, duplicates and dtype issues.",
    "Data Cleaning: remove duplicate reviews, convert dates, fill/impute NaNs, translate categories, IQR-cap outliers.",
    "Feature Engineering: compute delivery_days, is_late, freight_ratio, purchase time features, and the low_review target.",
    "EDA: 11 visualisations covering revenue trends, category performance, delivery time, review distribution, and correlations.",
    "Statistical Testing: Mann-Whitney U (delivery punctuality vs review score) and Chi-square + Cramer's V (is_late vs low_review).",
    "Customer Segmentation: RFM scoring, KMeans elbow + silhouette selection, segment profiling.",
    "Machine Learning: Logistic Regression (baseline), Random Forest, XGBoost; class_weight/scale_pos_weight for imbalance; RandomizedSearchCV tuning; SHAP explanations.",
    "App: Streamlit dashboard with Overview, Charts, Segments and Prediction pages.",
    "Report & README: python-docx report, GitHub-ready README.",
]:
    add_bullet(step)

# ══════════════════════════════════════════════════════════════════════════════
# 6. DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
add_title("6. Data Cleaning", level=1)
add_para(
    "The cleaning log below shows before/after row counts for each action. "
    "All raw tables had zero duplicate rows; the key cleaning operations were:"
)

clean_tbl = doc.add_table(rows=1, cols=4)
clean_tbl.style = 'Table Grid'
for i,h in enumerate(['Table','Action','Before','Removed']):
    clean_tbl.rows[0].cells[i].text = h
    clean_tbl.rows[0].cells[i].paragraphs[0].runs[0].bold = True

cleaning_data = [
    ('orders',   'drop_duplicates',               '99,441', '0'),
    ('items',    'drop_duplicates',               '112,650','0'),
    ('payments', 'drop_duplicates',               '103,886','0'),
    ('reviews',  'drop_duplicates',               '99,224', '0'),
    ('reviews',  'keep_latest_review_per_order',  '99,224', '551'),
    ('products', 'fill category NaN -> unknown',  '32,951', '0'),
    ('products', 'fill weight/dim NaN -> median', '32,951', '0'),
    ('master',   'IQR cap total_price [7.90, 997.82]',   '99,992', '0 rows; values capped'),
    ('master',   'IQR cap total_freight [0.00, 104.73]', '99,992', '0 rows; values capped'),
]
for row in cleaning_data:
    add_table_row(clean_tbl, row)
doc.add_paragraph()

add_para(
    "Master dataframe after all joins: 99,992 rows x 20 columns. "
    f"Delivered-only subset used for EDA and ML: {m['total_orders']:,} rows."
)

add_figure('00_missing_values.png', 'Figure 0: Missing Values (%) Before Cleaning', width=5.0)

# ══════════════════════════════════════════════════════════════════════════════
# 7. EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
add_title("7. Exploratory Data Analysis", level=1)

eda_figs = [
    ('01_monthly_revenue_orders.png',
     f"Figure 1: Monthly Revenue and Order Volume. Peak revenue R${m['top_category_rev']:,.0f} "
     f"in {m['peak_month']} coincides with Black Friday/Cyber Monday surge."),
    ('02_top10_categories_revenue.png',
     f"Figure 2: Top 10 Categories by Revenue. "
     f"{m['top_category']} leads with R${m['top_category_rev']:,.0f}."),
    ('03_orders_weekday_hour.png',
     f"Figure 3: Orders by Day of Week and Hour. Busiest day: {m['peak_day']}; "
     f"peak hour: {m['peak_hour']}:00."),
    ('04_payment_type.png',
     f"Figure 4: Payment Type Distribution. "
     f"{m['dominant_payment']} dominates at {m['dominant_payment_pct']:.1f}%."),
    ('05_review_score_dist.png',
     f"Figure 5: Review Score Distribution. "
     f"Average score: {m['avg_review_score']:.2f}. Low-review (<=2): {m['low_review_pct']:.1f}%."),
    ('06_delivery_time_dist.png',
     f"Figure 6: Delivery Time Distribution. "
     f"Mean={m['avg_delivery_days']:.1f}d, Median={m['median_delivery_days']:.1f}d."),
    ('07_late_vs_ontime.png',
     f"Figure 7: Late vs On-Time Deliveries. {m['late_pct']:.1f}% of orders arrived after the estimate."),
    ('08_delivery_by_state.png',
     f"Figure 8: Avg Delivery by State. Slowest: {m['worst_state']} ({m['worst_days']:.1f}d); "
     f"Fastest: {m['best_state']} ({m['best_days']:.1f}d)."),
    ('09_revenue_by_state.png',
     f"Figure 9: Revenue by State. Top state: {m['top_rev_state']} "
     f"(R${m['top_rev_state_rev']:,.0f})."),
    ('10_freight_ratio_review.png',
     f"Figure 10: Freight Ratio by Review Score. "
     f"Low-review median={m['med_fr_low']:.3f} vs high-review={m['med_fr_high']:.3f}."),
    ('11_correlation_heatmap.png',
     f"Figure 11: Correlation Heatmap. "
     f"is_late has correlation {m['corr_late_review']:.4f} with review_score."),
]

for fname, caption in eda_figs:
    add_figure(fname, caption, width=5.2)
    doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# 8. STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
add_title("8. Statistical Analysis", level=1)

add_para("Test 1 – Mann-Whitney U: Late vs On-Time Review Scores", bold=True)
add_para(
    f"H0: Review scores are equal for late and on-time deliveries.\n"
    f"H1: Late deliveries have lower review scores.\n"
    f"Both distributions are non-normal (Shapiro-Wilk p<<0.05 on n=5,000 subsample), "
    f"so the non-parametric Mann-Whitney U test is appropriate.\n"
    f"Result: U={m['mannwhitney_u']:,.0f}, p={m['mannwhitney_p']:.4e}, "
    f"Effect size r={m['effect_r']:.4f} (large effect, |r|>=0.3).\n"
    f"Conclusion: REJECT H0. Late deliveries score {m['late_mean_score']:.2f} vs "
    f"{m['ontime_mean_score']:.2f} on-time — a practically significant difference."
)

doc.add_paragraph()
add_para("Test 2 – Chi-square + Cramer's V: is_late vs low_review", bold=True)
add_para(
    f"Chi2={m['chi2']:.4f}, df=1, p={m['chi2_p']:.4e}.\n"
    f"Cramer's V={m['cramers_v']:.4f} (medium-large association).\n"
    f"Conclusion: There is a statistically significant and practically meaningful "
    f"association between late delivery and the probability of a low review. "
    f"Being late increases the low-review rate from ~9% to ~61%."
)

# ══════════════════════════════════════════════════════════════════════════════
# 9. CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
add_title("9. Customer Segmentation", level=1)

add_para(
    f"RFM analysis was performed on {m['num_customers']:,} customers with a snapshot date "
    f"of {m['rfm_snapshot']}. Because 96% of customers have exactly one order, frequency is "
    f"constant; segmentation is driven by recency and monetary value.\n\n"
    f"KMeans was run for k=2..8 using a 20,000-row stratified subsample. "
    f"The best k={m['best_k']} was selected by silhouette score ({m['best_silhouette']:.4f})."
)

seg_map = m.get('seg_map', {})
cluster_sizes = m.get('cluster_sizes', {})
if seg_map:
    seg_tbl = doc.add_table(rows=1, cols=3)
    seg_tbl.style = 'Table Grid'
    for i,h in enumerate(['Cluster','Segment Name','Customer Count']):
        seg_tbl.rows[0].cells[i].text = h
        seg_tbl.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    for cl, seg_name in seg_map.items():
        cnt = cluster_sizes.get(seg_name, 'N/A')
        add_table_row(seg_tbl, [cl, seg_name, str(cnt)])
    doc.add_paragraph()

add_figure('12_kmeans_elbow_silhouette.png',
           f"Figure 12: KMeans Elbow and Silhouette Curves. Best k={m['best_k']}.", width=5.5)
add_figure('13_rfm_clusters.png',
           'Figure 13: RFM Cluster Scatter (Recency vs Monetary).', width=5.2)
add_figure('14_segment_distribution.png',
           'Figure 14: Customer Count per Segment.', width=5.0)

# ══════════════════════════════════════════════════════════════════════════════
# 10. MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════
add_title("10. Machine Learning Model", level=1)

add_para("Setup", bold=True)
add_para(
    f"Target: low_review (review_score <= 2). Class balance: ~12.7% positive.\n"
    f"Train set: {m['ml_train_size']:,} rows | Test set: {m['ml_test_size']:,} rows "
    f"(stratified 80/20 split). class_weight='balanced' / scale_pos_weight used to "
    f"handle imbalance without SMOTE (faster, comparable performance).\n"
    f"No data leakage: review_score and all review columns excluded from features."
)

add_para("Model Comparison", bold=True)
mc = m.get('model_comparison', {})
tuned_m = m.get('tuned_metrics', {})
best_nm = m.get('best_model_name', '')

cmp_tbl = doc.add_table(rows=1, cols=6)
cmp_tbl.style = 'Table Grid'
for i, h in enumerate(['Model','Accuracy','Precision','Recall','F1','ROC-AUC']):
    cmp_tbl.rows[0].cells[i].text = h
    cmp_tbl.rows[0].cells[i].paragraphs[0].runs[0].bold = True

for name, vals in mc.items():
    add_table_row(cmp_tbl, [
        name,
        f"{vals.get('accuracy',0):.4f}",
        f"{vals.get('precision',0):.4f}",
        f"{vals.get('recall',0):.4f}",
        f"{vals.get('f1',0):.4f}",
        f"{vals.get('roc_auc',0):.4f}",
    ])

add_table_row(cmp_tbl, [
    f"Tuned {best_nm} (FINAL)",
    f"{tuned_m.get('accuracy',0):.4f}",
    f"{tuned_m.get('precision',0):.4f}",
    f"{tuned_m.get('recall',0):.4f}",
    f"{tuned_m.get('f1',0):.4f}",
    f"{tuned_m.get('roc_auc',0):.4f}",
], bold=True)
doc.add_paragraph()

add_para(
    "Why F1 and Recall matter more than Accuracy: The dataset is ~87% non-low. "
    "A naive classifier predicting 'not low' every time achieves 87% accuracy but zero recall. "
    "F1 balances precision and recall; high recall ensures at-risk orders are flagged for intervention."
)

best_params = m.get('best_params', {})
if best_params:
    add_para(f"Best hyperparameters ({best_nm}): {best_params}", italic=True)

top_feat = m.get('top_feature', 'N/A')
add_para(f"Most important feature: {top_feat}", bold=True)

add_figure('15_confusion_matrix.png',
           f"Figure 15: Confusion Matrix — {best_nm} (Tuned).", width=3.5)
add_figure('16_roc_curves.png',
           'Figure 16: ROC Curves — All Models.', width=5.2)
if os.path.exists(f'{FIG_DIR}/17_feature_importance.png'):
    add_figure('17_feature_importance.png',
               f"Figure 17: Top 20 Feature Importances — {best_nm}.", width=5.2)
if os.path.exists(f'{FIG_DIR}/18_shap_summary.png'):
    add_figure('18_shap_summary.png',
               f"Figure 18: SHAP Summary — {best_nm}.", width=5.2)

# ══════════════════════════════════════════════════════════════════════════════
# 11. STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════
add_title("11. Streamlit App", level=1)
add_para(
    "The interactive dashboard (app.py) is built with Streamlit 1.60.0 and has four tabs:"
)
for page in [
    "Overview: KPI cards (total revenue, orders, avg review score, % late, avg delivery days) "
    "with sidebar filters (year, state, category). Uses st.cache_data for fast reloads.",
    "Charts: Plotly charts for revenue trend, top categories, delivery by state, "
    "review score distribution, freight ratio histogram.",
    "Customer Segments: RFM cluster summary table, bar chart and scatter plot; "
    "segment descriptions with actionable advice.",
    "Prediction: Form with all 15 model features. Submitting calls the saved "
    "models/best_model.joblib and returns the low-review probability with a "
    "Plotly gauge chart and a colour-coded risk label.",
]:
    add_bullet(page)
add_para(
    "Run command: streamlit run app.py\n"
    "Data files required: data/delivered.csv, data/rfm_segments.csv, "
    "data/metrics_fg.json, models/best_model.joblib"
)

# ══════════════════════════════════════════════════════════════════════════════
# 12. BUSINESS RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
add_title("12. Business Recommendations", level=1)

recs = [
    (
        "Fix Last-Mile Logistics in Slow States",
        f"{m['late_pct']:.1f}% of orders arrive late. Late deliveries score {m['late_mean_score']:.2f} "
        f"vs {m['ontime_mean_score']:.2f} for on-time (r={m['effect_r']:.2f}, large effect). "
        f"State {m['worst_state']} averages {m['worst_days']:.0f} days vs the national mean of "
        f"{m['avg_delivery_days']:.0f} days.",
        "Establish regional fulfilment hubs in the North/Northeast or negotiate SLA-based "
        "carrier contracts with performance penalties for states exceeding 20-day delivery."
    ),
    (
        "Pre-Position Inventory for November Peak",
        f"Revenue peaked at R${m['top_category_rev']:,.0f} in {m['peak_month']}. Late deliveries "
        f"surge during high-volume periods.",
        "Begin seller capacity planning in September-October; monitor delivery SLAs in real-time "
        "during the November Black Friday window."
    ),
    (
        "Cap High Freight-to-Price Ratios",
        f"Low-review orders have median freight ratio {m['med_fr_low']:.3f} vs "
        f"{m['med_fr_high']:.3f} for high-review orders.",
        "Alert customers at checkout when freight exceeds 25% of item value. Introduce a "
        "'freight guarantee' badge for qualifying sellers."
    ),
    (
        "Prioritise Top Revenue Categories",
        f"{m['top_category']} generates R${m['top_category_rev']:,.0f} in revenue. "
        f"The top 10 categories drive the majority of platform GMV.",
        "Implement category-specific seller quality scoring. Give top-10 category sellers "
        "priority placement and dedicated account management."
    ),
    (
        "Win Back Dormant Customers",
        f"The Lapsed Buyers segment (~39,000 customers) has high recency (389 days) "
        "but previously spent a comparable amount to active customers.",
        "Launch personalised win-back campaigns (voucher + recommended product) targeting "
        "customers inactive for 6-12 months."
    ),
]

for i, (title, evidence, action) in enumerate(recs, 1):
    add_para(f"Recommendation {i}: {title}", bold=True)
    add_para(f"Evidence: {evidence}", italic=True)
    add_para(f"Action: {action}")
    doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# 13. LIMITATIONS & FUTURE SCOPE
# ══════════════════════════════════════════════════════════════════════════════
add_title("13. Limitations & Future Scope", level=1)
add_para("Limitations", bold=True)
for lim in [
    "Dataset covers 2016-2018 only; post-pandemic delivery trends may differ significantly.",
    "Review comment text (NLP sentiment) was not utilised — could improve model accuracy.",
    "Geolocation data was not incorporated into delivery-time modelling.",
    "96% of customers have frequency=1; RFM segmentation is driven by Recency and Monetary only.",
    "This is a post-delivery model — delivery outcome features (actual days, delay) are only "
    "known after the order arrives. It cannot score orders before dispatch.",
    "ML performance is moderate (F1=0.46, AUC=0.77); suitable as a risk-prioritisation tool, "
    "not a definitive classifier.",
]:
    add_bullet(lim)

add_para("Future Scope", bold=True)
for fw in [
    "Real-time prediction API (FastAPI) wrapping models/best_model.joblib.",
    "NLP sentiment scoring on review_comment_message.",
    "Demand forecasting per category per state (ARIMA / Prophet).",
    "Geo-spatial heatmaps of delivery delays using the geolocation table.",
    "Advanced class-imbalance techniques and calibrated probability outputs.",
]:
    add_bullet(fw)

# ══════════════════════════════════════════════════════════════════════════════
# 14. CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
add_title("14. Conclusion", level=1)
add_para(
    f"This project demonstrated that delivery punctuality is the single most powerful driver "
    f"of customer review quality on the Olist platform. A large-effect Mann-Whitney U test "
    f"(r={m['effect_r']:.2f}, p < 0.001) and Chi-square (V={m['cramers_v']:.4f}) both confirm "
    f"the link between late delivery and low reviews. A tuned {m['best_model_name']} "
    f"post-delivery model achieved ROC-AUC={m['tuned_metrics']['roc_auc']:.4f} and "
    f"F1={m['tuned_metrics']['f1']:.4f} — moderate performance, appropriate as a "
    f"risk-prioritisation tool for operations teams. "
    f"RFM segmentation identified {m['best_k']} customer clusters (High-Value Buyers, "
    f"Recent Buyers, Lapsed Buyers), and five evidence-backed recommendations were produced. "
    f"All results are reproducible by running the four part_*.py scripts in order."
)

# ══════════════════════════════════════════════════════════════════════════════
# 15. REFERENCES
# ══════════════════════════════════════════════════════════════════════════════
add_title("15. References", level=1)
refs = [
    "Olist. (2018). Brazilian E-Commerce Public Dataset by Olist. Kaggle. "
    "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.",
    "Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.",
    "Mann, H. B., & Whitney, D. R. (1947). On a Test of Whether One of Two Variables is "
    "Stochastically Larger than the Other. Annals of Mathematical Statistics.",
    "RFM Analysis for Customer Segmentation. (Various authors). "
    "https://en.wikipedia.org/wiki/RFM_(market_research)",
]
for r in refs:
    add_bullet(r)

# ══════════════════════════════════════════════════════════════════════════════
# 16. USE OF IBM BOB AI TOOL
# ══════════════════════════════════════════════════════════════════════════════
add_title("16. Use of IBM Bob AI Tool", level=1)
add_para(
    "IBM Bob (an AI coding assistant) was used as a collaborative tool throughout "
    "this project in the following ways:"
)
for use in [
    "Scaffolding the project structure (folder layout, file names, requirements.txt).",
    "Writing and iteratively debugging the four Python scripts (part_ae.py, part_fg.py, "
    "part_h.py, part_i.py) that constitute the notebook logic.",
    "Diagnosing and fixing runtime errors including Unicode encoding issues on Windows, "
    "pandas API changes in v3.0.5 (FutureWarning for fillna), and XGBoost parameter "
    "compatibility.",
    "Generating the Streamlit app (app.py) including the prediction form and Plotly charts.",
    "Drafting this project report (build_report.py) and the README.md.",
    "All analytical decisions, hypotheses, interpretations and business recommendations "
    "were reviewed and validated by the author (Siddu Varikuppala).",
]:
    add_bullet(use)

doc.add_paragraph()
add_para(
    "[PLACEHOLDER — Insert IBM Bob screenshots here]",
    bold=True, italic=True
)
add_para(
    "Please add screenshots of key Bob interactions (e.g., debugging session, "
    "code generation, chart building) in this section before final submission.",
    italic=True
)

# ── Footer note ───────────────────────────────────────────────────────────────
add_page_break()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run(
    f"Siddu Varikuppala | AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 | BharatCares\n"
    f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
run.font.size = Pt(10)
run.font.color.rgb = RGBColor(0x57, 0x60, 0x6a)
run.italic = True

# ── Save ──────────────────────────────────────────────────────────────────────
doc.save(REPORT_PATH)
print(f"Saved: {REPORT_PATH}")
print(f"  Sections: {len(doc.sections)}")
print(f"  Paragraphs: {len(doc.paragraphs)}")
