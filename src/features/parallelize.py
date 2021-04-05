import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import Pool


def apply_parallel(df, function_name, num_partitions=40, num_cores=40, **kwargs):
    df_split = np.array_split(df, num_partitions)
    with Pool(num_cores) as pool:
        df = pd.concat(pool.map(partial(function_name, **kwargs), df_split), sort=False)
    return df
