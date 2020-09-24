import numpy as np

def HighlightChanges(df):
    df_highlight = df.copy(deep=True)
    for f in range(df.shape[1]):
        ind = np.where(df.iloc[:,f] == df.iloc[0,f])[0]
        df_highlight.iloc[ind[1:],f] = '_'

    return df_highlight