import pandas as pd
from upsetplot import from_memberships, UpSet
import matplotlib.pyplot as plt
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate upset plot for feature selection methods")
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

# Extract top 100 features (files are already ranked)
elastic_net_features = set(elastic_net["feature"].head(TOP_N))
lightgbm_features = set(lightgbm["feature"].head(TOP_N))
ntk_features = set(ntk["ICD10_3Digits"].head(TOP_N))
recurrence_features = set(recurrence["ICD10_3Digits"].head(TOP_N))

# Create method-to-features mapping
methods = {
    "Elastic Net": elastic_net_features,
    "LightGBM": lightgbm_features,
    "NTK": ntk_features,
    "Recurrence Enrichment": recurrence_features,
}

# Build membership list for each feature
all_features = set().union(*methods.values())
memberships = []
for feature in all_features:
    member_of = tuple(name for name, feat_set in methods.items() if feature in feat_set)
    memberships.append(member_of)

# Create upset data
upset_data = from_memberships(memberships)

# Plot
fig = plt.figure(figsize=(12, 8))
upset = UpSet(upset_data, subset_size="count", show_counts=True, sort_by="cardinality")
upset.plot(fig=fig)
plt.suptitle(f"Overlap of Top {TOP_N} Diagnosis Features Across Selection Methods", fontsize=14)

# Save and show
plt.savefig(f"{DATA_DIR}/upset_plot_top{TOP_N}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{DATA_DIR}/upset_plot_top{TOP_N}.pdf", bbox_inches="tight")
print(f"Saved: upset_plot_top{TOP_N}.png and upset_plot_top{TOP_N}.pdf")

# Find features selected by 3 or more methods
print("\n" + "="*60)
print(f"Features selected by 3 or more methods (top {TOP_N} each):")
print("="*60)

feature_counts = {}
for feature in all_features:
    count = sum(1 for feat_set in methods.values() if feature in feat_set)
    if count >= 3:
        feature_counts[feature] = count

# Sort by count (descending) then by feature name
sorted_features = sorted(feature_counts.items(), key=lambda x: (-x[1], x[0]))

# Load diagnosis descriptions for display (from all sources with descriptions)
desc_map = {}
for _, row in recurrence.iterrows():
    desc_map[row["ICD10_3Digits"]] = row.get("DIAGNOSIS_DESCRIPTION", "")
for _, row in ntk.iterrows():
    desc_map[row["ICD10_3Digits"]] = row.get("DIAGNOSIS_DESCRIPTION", "")
for _, row in lightgbm.iterrows():
    desc_map[row["feature"]] = row.get("DIAGNOSIS_DESCRIPTION", "")

print(f"\nTotal: {len(sorted_features)} features\n")
print(f"{'Feature':<10} {'Methods':<10} {'Description'}")
print("-"*80)
for feature, count in sorted_features:
    desc = desc_map.get(feature, "")[:50]
    in_methods = [name for name, feat_set in methods.items() if feature in feat_set]
    print(f"{feature:<10} {count:<10} {desc}")
    print(f"           Methods: {', '.join(in_methods)}")
    print()

# Save to CSV
results_df = pd.DataFrame([
    {
        "feature": f,
        "n_methods": c,
        "methods": ", ".join([name for name, feat_set in methods.items() if f in feat_set]),
        "description": desc_map.get(f, "")
    }
    for f, c in sorted_features
])
results_df.to_csv(f"{DATA_DIR}/features_3plus_methods_top{TOP_N}.csv", index=False)
print(f"\nSaved feature list to: features_3plus_methods_top{TOP_N}.csv")

plt.show()
