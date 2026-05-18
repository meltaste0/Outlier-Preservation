import pandas as pd
df = pd.read_csv('arrhythmia.data', header=None)
df = df.replace('?', pd.NA).dropna()
df.to_csv('arrhythmia_clean.data', index=False, header=False)