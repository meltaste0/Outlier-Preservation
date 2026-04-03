import pandas as pd

# df = pd.read_csv("Datasets\\Heart Disease\\heart_disease_health_indicators_BRFSS2015.csv")
# # Move class column to the end
# class_col = "HeartDiseaseorAttack"
# cols = [c for c in df.columns if c != class_col] + [class_col]
# df = df[cols]
# df.to_csv("Datasets\\Heart Disease\\heart_disease_cleaned.csv", index=False)
df = pd.read_csv("models\\CTAB-GAN-Plus\\Fake_Datasets\\Heart_Disease\\Heart_Disease_fake_0.csv")
print(df.shape)
df = pd.read_csv("models\\CTGAN\\Fake_Datasets\\synthetic_heart_disease_ctgan.csv")
print(df.shape)