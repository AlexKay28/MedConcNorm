import numpy as np
import pandas as pd

class FeatureGenerator:

    __sources = {
        'psytar': None,
        'mrconso': None,
        'cadec': None
    }

    def __init__(self, data: pd.DataFrame, n_jobs=5):
        self.data = data
        self.n_jobs = n_jobs

    def get_terms_from_dict(self, dict_name='psytar') -> pd.DataFrame:
        pass  #TODO

    def get_number_of_terms_from_dict(self, dict_name='psytar') -> pd.DataFrame:
        pass  #TODO

    def is_top_adrs(self, list_of_tops: list) -> pd.DataFrame:
        pass  #TODO
