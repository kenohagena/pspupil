import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

runs = {
    '001': {'Baseline': ['000', '001', '002'],
            'Followup': ['001', '002', '004']},
    '002': {'Baseline': ['000', '001', '002', '003'],  # welche 3
            'Followup': None},
    '007': {'Baseline': ['000', '001', '002'],
            'Followup': ['000', '001', '002']},
    '010': {'Baseline': ['000', '001', '002'],
            'Followup': ['004', '005', '006']},
    '012': {'Baseline': ['000', '001', '002'],
            'Followup': ['000', '001', '002']},
    '014': {'Baseline': ['000', '001', '002'],
            'Followup': None},
    '018': {'Baseline': ['001', '003', '004'],
            'Followup': ['000', '001', '002']},
    '019': {'Baseline': ['000', '001', '002'],
            'Followup': ['001', '002', '003']},
    '020': {'Baseline': ['000', '001', '002'],
            'Followup': ['000', '001', '002']},
    '021': {'Baseline': ['001', '002', '003'],
            'Followup': None},
    '023': {'Baseline': ['000', '001', '003'],
            'Followup': None},
    '025': {'Baseline': ['000', '001', '002'],
            'Followup': None},
    '026': {'Baseline': ['000', '001', '002'],
            'Followup': None},
    '003': {'Baseline': ['000', '001', '002'],
            'Followup': ['000', '001', '002']},
    '006': {'Baseline': ['000', '001', '004'],
            'Followup': ['000', '001', '002']},
    '009': {'Baseline': ['000', '001', '002'],
            'Followup': None},
    '013': {'Baseline': ['000', '001', '002'],
            'Followup': ['000', '001', '002']},
    '016': {'Baseline': ['000', '001', '002'],
            'Followup': ['001', '002', '003']},
    '022': {'Baseline': ['000', '001', '002'],
            'Followup': None},
    '024': {'Baseline': ['000', '001', '002'],
            'Followup': None}
}


rs = {
    '001': True,
    '002': True,
    '007': True,
    '010': True,
    '012': True,
    '014': True,
    '018': True,
    '019': True,
    '020': True,
    '021': True,
    '023': True,
    '025': True,
    '026': True,
    '003': False,
    '006': False,
    '009': False,
    '013': False,
    '016': False,
    '022': False,
    '024': False
}
