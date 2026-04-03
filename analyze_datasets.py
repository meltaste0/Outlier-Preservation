import pandas as pd

# ARRHYTHMIA


# with open("arrhythmia.data", newline="", encoding="utf-8") as f:
    # reader = csv.reader(f)
    # for row in reader:
    #     if not row:
    #         continue
    #     last_value =row[-1].strip()
    #     dict[last_value]=dict.get(last_value,0)+1
    # data = pd.read_csv("arrhythmia.data", header=None)
    # print(data.head())
    # print(data.shape)
    # for key, value in dict.items():
    #     print(f"{key}: {value}")


#------------------------------------

#THYROID

# train_data = pd.read_csv("Thyroid/ann-train.data", sep=r"\s+", header=None)
# train_class_counts = train_data.iloc[:, -1].astype(str).str.strip().value_counts().sort_index()

# data = pd.read_csv("Thyroid/ann-test.data", sep=r"\s+", header=None)
# test_class_counts = data.iloc[:, -1].astype(str).str.strip().value_counts().sort_index()

# print(data.head())
# print(data.shape)

# print("\nThyroid train class counts:")
# for key, value in train_class_counts.items():
#     print(f"{key}: {value}")

# print("\nThyroid test class counts:")
# for key, value in test_class_counts.items():
#     print(f"{key}: {value}")


#--------------------------------------------



# BREAST CANCER
# data = pd.read_csv("Breast Cancer/Breast_Cancer.csv")
# data.columns = data.columns.str.strip()

# a_stage_counts = (
#     data["A Stage"]
#     .astype(str)
#     .str.strip()
#     .str.lower()
#     .value_counts()
#     .reindex(["regional", "distant"], fill_value=0)
# )

# print("A Stage counts:")
# for label, count in a_stage_counts.items():
#     print(f"{label}: {count}")

#------------------------------------

#THYROID SYNTHETIC CTGAN

# class_counts = {}
# with open("synthetic_thyroid.csv", encoding="utf-8") as f:
#     for line in f:
#         parts = line.split(',')
#         if not parts:
#             continue
#         label = parts[-1].strip()
#         if not label or label.lower() == "class":
#             continue
#         class_counts[label] = class_counts.get(label, 0) + 1

# data = pd.read_csv("synthetic_thyroid.csv", sep=r"\s+", header=None)
# print(data.head())
# print(data.shape)
# for key, value in sorted(class_counts.items()):
#     print(f"{key}: {value}")


#------------------------------------

# SYNTHETIC ARRHYTHMIA

data = pd.read_csv("synthetic_arrhythmia.csv")
print(data.head())
print(data.shape)

arrhythmia_class_counts = (
    pd.to_numeric(data.iloc[:, -1], errors="coerce")
    .dropna()
    .astype(int)
    .value_counts()
    .sort_index()
    .reindex(range(1, 17), fill_value=0)
)

print("\nSynthetic arrhythmia class counts (1-16):")
for key, value in arrhythmia_class_counts.items():
    print(f"{key}: {value}")
