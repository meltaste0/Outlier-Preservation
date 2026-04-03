import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

# The thyroid training file is whitespace-delimited (not CSV).
# If loaded with comma defaults, each row becomes one unique string value,
# which makes CTGAN expand into thousands of columns and train very slowly.
data = pd.read_csv('Thyroid/ann-train.data', sep=r'\s+', header=None)

# This dataset has 21 features plus 1 target class.
data.columns = [f'Feature_{i}' for i in range(21)] + ['Class']
data['Class'] = data['Class'].astype(str)

metadata = Metadata.detect_from_dataframe(data)

synthesizer = CTGANSynthesizer(
     metadata,
     epochs=50,
     enforce_rounding=True,
     verbose=True,
)
synthesizer.fit(data)
synthetic_data = synthesizer.sample(num_rows=3772)
with open("synthetic_thyroid.csv","w", encoding="utf-8") as f:
     synthetic_data.to_csv(f, index=False)