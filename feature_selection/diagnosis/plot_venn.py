import pandas as pd
import matplotlib.pyplot as plt
import argparse
from supervenn import supervenn

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate Venn diagram for feature selection methods")
parser.add_argument("-n", "--top_n", type=int, default=100, help="Number of top features to use (default: 100)")
args = parser.parse_args()

# Configuration
TOP_N = args.top_n
DATA_DIR = "/home/zihan/pcori_project/final_data/feature_selection/diagnosis"

# Load data and extract top N features from each method
elastic_net = pd.read_csv(f"{DATA_DIR}/elastic_net_full.csv")
lightgbm = pd.read_csv(f"{DATA_DIR}/lightgbm_full.csv")
ntk = pd.read_csv(f"{DATA_DIR}/ntk_full.csv")
recurrence = pd.read_csv(f"{DATA_DIR}/recurrence_enrichment_full.csv")

# Extract top N features (files are already ranked)
elastic_net_features = set(elastic_net["feature"].head(TOP_N))
lightgbm_features = set(lightgbm["feature"].head(TOP_N))
ntk_features = set(ntk["ICD10_3Digits"].head(TOP_N))
recurrence_features = set(recurrence["ICD10_3Digits"].head(TOP_N))

# Build description map
desc_map = {}
for _, row in recurrence.iterrows():
    desc_map[row["ICD10_3Digits"]] = row.get("DIAGNOSIS_DESCRIPTION", "")
for _, row in ntk.iterrows():
    desc_map[row["ICD10_3Digits"]] = row.get("DIAGNOSIS_DESCRIPTION", "")
for _, row in lightgbm.iterrows():
    desc_map[row["feature"]] = row.get("DIAGNOSIS_DESCRIPTION", "")

# Create lists for supervenn
sets = [elastic_net_features, lightgbm_features, ntk_features, recurrence_features]
labels = ["Elastic Net", "LightGBM", "NTK", "Recurrence Enrichment"]

# Plot with supervenn - shows actual elements
fig, ax = plt.subplots(figsize=(20, 10))
supervenn(
    sets,
    labels,
    side_plots=True,
    chunks_ordering='size',
    sets_ordering=None,
    reverse_chunks_order=False,
    max_bruteforce_size=2**16,
    seeds=0,
    noise_prob=0,
    ax=ax
)
plt.title(f"Feature Overlap: Top {TOP_N} Diagnosis Features Across Selection Methods", fontsize=14, pad=20)

# Save supervenn plot
plt.savefig(f"{DATA_DIR}/venn_plot_top{TOP_N}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{DATA_DIR}/venn_plot_top{TOP_N}.pdf", bbox_inches="tight")
print(f"Saved: venn_plot_top{TOP_N}.png and venn_plot_top{TOP_N}.pdf")
plt.show()

# Also create a detailed text figure showing ICD codes in each intersection
from itertools import combinations

method_names = ["Elastic Net", "LightGBM", "NTK", "Recurrence Enrichment"]
method_sets = [elastic_net_features, lightgbm_features, ntk_features, recurrence_features]
methods = dict(zip(method_names, method_sets))

def get_exact_intersection(include_methods, exclude_methods, methods_dict):
    """Get features that are in ALL include_methods and NONE of exclude_methods"""
    if not include_methods:
        return set()
    result = set.intersection(*[methods_dict[m] for m in include_methods])
    for m in exclude_methods:
        result = result - methods_dict[m]
    return result

# Generate all regions
all_regions = []
for r in range(1, 5):
    for combo in combinations(method_names, r):
        include = set(combo)
        exclude = set(method_names) - include
        features = get_exact_intersection(include, exclude, methods)
        if features:
            all_regions.append({
                "methods": combo,
                "n_methods": len(combo),
                "features": features,
                "count": len(features)
            })

all_regions.sort(key=lambda x: (-x["n_methods"], -x["count"]))

# Create a text-based figure showing all ICD codes with descriptions
fig2, ax2 = plt.subplots(figsize=(24, 16))
ax2.axis('off')

y_pos = 0.98
line_height = 0.018

text_content = f"Top {TOP_N} Diagnosis Features by Intersection\n"
text_content += "=" * 80 + "\n\n"

for region in all_regions:
    methods_str = " ∩ ".join(region["methods"])
    text_content += f"[{region['n_methods']} methods] {methods_str} (n={region['count']})\n"
    text_content += "-" * 60 + "\n"

    for feat in sorted(region["features"]):
        desc = desc_map.get(feat, "")[:60]  # Truncate long descriptions
        text_content += f"  {feat}: {desc}\n"
    text_content += "\n"

ax2.text(0.02, 0.98, text_content, transform=ax2.transAxes,
         fontsize=8, fontfamily='monospace', verticalalignment='top')

plt.title(f"ICD Codes and Descriptions - Top {TOP_N} Features", fontsize=14)
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/venn_details_top{TOP_N}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{DATA_DIR}/venn_details_top{TOP_N}.pdf", bbox_inches="tight")
print(f"Saved: venn_details_top{TOP_N}.png and venn_details_top{TOP_N}.pdf")
plt.show()

# Save CSV
csv_rows = []
for region in all_regions:
    methods_str = " & ".join(region["methods"])
    for feat in sorted(region["features"]):
        csv_rows.append({
            "feature": feat,
            "description": desc_map.get(feat, ""),
            "n_methods": region["n_methods"],
            "methods": methods_str
        })

details_df = pd.DataFrame(csv_rows)
details_df.to_csv(f"{DATA_DIR}/venn_details_top{TOP_N}.csv", index=False)
print(f"Saved CSV: venn_details_top{TOP_N}.csv")
