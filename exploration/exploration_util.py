import numpy as np
import pandas as pd


def remove_outliers(df, column_names):
    data = pd.DataFrame(df)

    for column_name in column_names:
        q1 = np.percentile(data[column_name], 25)
        q3 = np.percentile(data[column_name], 75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr

        length_before = len(data)
        data = data.drop((data[data[column_name] >= upper].index | data[data[column_name] <= lower].index), axis=0)

        print('Removed ' + str(length_before - len(data)) + ' outliers of ' + column_name + '.')

    return data
