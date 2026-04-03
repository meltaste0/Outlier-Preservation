import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

data = pd.read_csv('Arrhythmia/arrhythmia.data', sep=r',', header=None, na_values=['?'])

# This dataset has 279 features plus 1 target class (280 columns total)
data.columns = [f'Feature_{i}' for i in range(279)] + ['Class']
data['Class'] = data['Class'].astype(str)

metadata = Metadata.detect_from_dataframe(data)

synthesizer = CTGANSynthesizer(
     metadata,
     epochs=50,
     enforce_rounding=True,
     verbose=True,
)
synthesizer.fit(data)
synthetic_data = synthesizer.sample(num_rows=len(data))
with open("synthetic_arrhythmia.csv","w", encoding="utf-8") as f:
     synthetic_data.to_csv(f, index=False)