import pandas as pd

# found the gem csv file from "https://www.nber.org/research/data/icd-9-cm-and-icd-10-cm-and-icd-10-pcs-crosswalk-or-general-equivalence-mappings"
# download via "https://data.nber.org/gem/icd10cmtoicd9gem.csv"
gem_df = pd.read_csv('icd10cmtoicd9gem.csv')
print(gem_df.info())
gem_df = gem_df.rename(columns={'icd10cm': 'icd10'})


# found the description file from "https://www.cms.gov/medicare/coordination-benefits-recovery/overview/icd-code-lists"
# download via "https://www.cms.gov/files/document/valid-icd-10-list.xlsx"
desc_df = pd.read_excel('section111validicd10-jan2025_0.xlsx',
                         usecols=['CODE', 'LONG DESCRIPTION (VALID ICD-10 FY2025)'])
print(desc_df.info())
desc_df = desc_df.rename(columns={'CODE': 'icd10', 'LONG DESCRIPTION (VALID ICD-10 FY2025)': 'description'})


# perform a left join to keep all ICD-9 codes and add description of ICD-10 where available
merged_df = pd.merge(
    gem_df,
    desc_df[['icd10', 'description']],
    on='icd10',
    how='left'
)

# number of missing description in merged_df
a = (merged_df['description'].isnull()).sum()
print(f"Missing number of description: {a}")

blank_rate = round((a / merged_df.shape[0]), 4) * 100
print(f"The blanking rate of the description of ICD-10: {blank_rate}%")


# found from "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2025-Update/"
# download via "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2025-Update/Code-desciptions-April-2025.zip"
with open("icd10cm-order-April-2025.txt") as file:
    lines = file.readlines()

# extract the ICD-10 codes and their desciption from the txt file
codes_list = []
desc_list = []
for i in range(len(lines)):
    codes = lines[i][6:14].strip()
    codes_list.append(codes)

    des = lines[i][77:-1].strip()
    desc_list.append(des)

data = list(zip(codes_list, desc_list))
supply_df = pd.DataFrame(data, columns=['ICD10', 'Description'])

# extract ICD-10 codes without description in merged_df
null_desc = merged_df["description"].isnull()

copy_merged = merged_df.copy()

for idx in merged_df[null_desc].index:
    icd10_code = merged_df.loc[idx, "icd10"]
    
    matching_desc = supply_df[supply_df["ICD10"] == icd10_code]["Description"]
    
    if not matching_desc.empty:
        copy_merged.loc[idx, "description"] = matching_desc.iloc[0]

# number of missing description in supplemented dataframe
b = (copy_merged["description"].isnull()).sum()
print(f"Missing number of description: {b}")
blank_rate = round((b / copy_merged.shape[0]), 4) * 100
print(f"The blanking rate of the description of ICD-10: {blank_rate}%")
print(f"The supplement quantity: {a - b}")

# reorder the columns
copy_merged = copy_merged[['icd9cm', 'icd10', "description", 'flags', 'approximate', 'no_map', 'combination', 'scenario', 'choice_list'] ]
print(copy_merged.info())

# manually enter the still missing descriptions
still_missing = copy_merged["description"].isnull()
print(copy_merged[still_missing])

### CHECKED: `still_missing` are deleted codes
# remove deteled codes
copy_merged = copy_merged.drop(copy_merged[still_missing].index).reset_index(drop=True)

c = (copy_merged["description"].isnull()).sum()
print(f"Missing number of description: {c}")
blank_rate = round((c / copy_merged.shape[0]), 4) * 100
print(f"The blanking rate of the description of ICD-10: {blank_rate}%")
print(copy_merged.info())


# export a csv file including ICD-9cm, ICD-10, description, etc
copy_merged.to_csv('ICD9cm_ICD10_description.csv', index = True) 


















