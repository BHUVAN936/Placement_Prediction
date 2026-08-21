import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

df = pd.DataFrame({
    "Age" : [25,30,22,35,28],
    "Salary(in K)" : [50,np.nan,45,70,60],
    "Experience" : [2,5,1,7,np.nan]
})
print(df)

imputer = KNNImputer(n_neighbors=1)
df_imputed = pd.DataFrame(imputer.fit_transform(df),columns=df.columns)
print('\nAfter KNN Imputation (K=1):')
print(df_imputed.round(2))