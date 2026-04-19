import numpy as np
import pandas as pd
import pyedflib
import neurokit2 as nk
from scipy.signal import welch
df = pd.read_csv("sleep_hrv_dataset.csv")
df[["frac_W","frac_N1","frac_N2","frac_N3","frac_R"]].sum(axis=1).describe()
