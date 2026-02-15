import numpy as np

def build_sequences(df, window=14):
    X, y = [], []

    features = [
        'Crop','Temperature_C','pH','VFA_mgL',
        'VS_percent','CN_Ratio','Lignin_percent'
    ]

    for run in df['Run_ID'].unique():
        run_df = df[df['Run_ID'] == run].sort_values('Day')

        for i in range(len(run_df) - window):
            X.append(run_df[features].iloc[i:i+window].values)
            y.append(run_df['Daily_Methane_m3'].iloc[i+window])

    return np.array(X), np.array(y)
