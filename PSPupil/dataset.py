import pandas as pd


# 1. subjects, sessions and runs for PSP patients
runs = {
    '001': {'Baseline': ['000', '001', '002'],
            'Followup': ['001', '002', '004']},
    '002': {'Baseline': ['001', '002', '003'],  # welche 3
            'Followup': None},
    '007': {'Baseline': ['000', '001', '002'],
            'Followup': ['000', '001', '002']},
    '010': {'Baseline': ['000', '001', '002'],
            'Followup': ['004', '005', '006']},
    '012': {'Baseline': ['000', '001', '002'],
            'Followup': ['000', '001', '002']},
    '013': {'Baseline': ['000', '001', '002'],
            'Followup': None},
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
    '016': {'Baseline': ['000', '001', '002'],
            'Followup': ['001', '002', '003']},
    '022': {'Baseline': ['000', '001', '002'],
            'Followup': None},
    '024': {'Baseline': ['000', '001', '002'],
            'Followup': None}
}

#1.1. New PSP subjects (i.e. Molly Zeitschel data)
runs_new_1 = {
    '027': {'Baseline': ['000', '001', '002'],
            'Followup': None},
    '028': {'Baseline': ['000', '001', '002'],
            'Followup': None}
}

runs_new_2 = {
    '030': {'Baseline': ['002', '003', '004'],
            'Followup': None},
    '031': {'Baseline': ['002', '003', '004'],
            'Followup': None},
    '032': {'Baseline': ['002', '003', '004'],
            'Followup': None},
    '033': {'Baseline': ['002', '003', '004'],
            'Followup': None},
    '034': {'Baseline': ['003', '004', '005'],
            'Followup': None},
    '037': {'Baseline': ['002', '003', '004'],
            'Followup': None},
    '038': {'Baseline': ['002', '003', '004'],
            'Followup': None},
    '039': {'Baseline': ['002', '003', '004'],
            'Followup': None},
    '040': {'Baseline': ['001', '002', '003'],
            'Followup': None},
    '041': {'Baseline': ['002', '003', '004'],
            'Followup': None},
    '042': {'Baseline': ['002', '003', '004'],
            'Followup': None},
    '045': {'Baseline': ['003', '004', '005'],
            'Followup': None},
    '046': {'Baseline': ['002', '003', '004'],
            'Followup': None},
    '047': {'Baseline': ['003', '004', '005'],
            'Followup': None}}

# 2. list of subjects that have a follow up session
follow = ['001', '007', '010', '012', '018', '019', '020', '003', '006', '016']

# 3. patients that have followup data with according runs
follow_runs = {
    '001': {'Baseline': ['000', '001', '002'],
            'Followup': ['001', '002', '004']},
    '007': {'Baseline': ['000', '001', '002'],
            'Followup': ['000', '001', '002']},
    '010': {'Baseline': ['000', '001', '002'],
            'Followup': ['004', '005', '006']},
    '012': {'Baseline': ['000', '001', '002'],
            'Followup': ['000', '001', '002']},
    '018': {'Baseline': ['001', '003', '004'],
            'Followup': ['000', '001', '002']},
    '019': {'Baseline': ['000', '001', '002'],
            'Followup': ['001', '002', '003']},
    '020': {'Baseline': ['000', '001', '002'],
            'Followup': ['000', '001', '002']},
    '003': {'Baseline': ['000', '001', '002'],
            'Followup': ['000', '001', '002']},
    '006': {'Baseline': ['000', '001', '004'],
            'Followup': ['000', '001', '002']},

    '016': {'Baseline': ['000', '001', '002'],
            'Followup': ['001', '002', '003']}

}

#4. control subjects with according runs
c_runs_g = {'02': ['000', '001', '002'],
          '03': ['000', '001', '002'],
          '04': ['000', '001', '002'],
          '05': ['000', '001', '002'],
          '06': ['001', '002', '003'],
          '07': ['000', '001', '002'],
          '08': ['000', '001', '002'],
          '09': ['000', '001', '002'],
          '10': ['000', '001', '002'],
          '11': ['000', '001', '002'],
          '12': ['000', '001', '002'],
          '13': ['000', '001', '002']}

c_runs_m ={
          '001': ['003', '004', '005'],
          '002': ['002', '003', '004'],
          '003': ['002', '003', '004'],
          '004': ['002', '003', '004'],
          '005': ['002', '003', '004'],
          '006': ['002', '003', '004'],
          '007': ['002', '003', '004'],
          '008': ['002', '004', '005'],
          '009': ['002', '003', '004'],
          '010': ['002', '003', '004'],
          '011': ['003', '004', '005'],
          '012': ['002', '003', '004'],
          '013': ['002', '003', '004'],
          '014': ['002', '003', '004'],
          '015': ['002', '003', '004'],
          '016': ['002', '003', '004'],
          '017': ['002', '003', '004'],
          '018': ['002', '003', '004'],
          '019': ['003', '004', '005']}


#5. list of patients that are grouped ath richardson-steel subtype
RS = ['001',
      '002',
      '007',
      '010',
      '012',
      '014',
      '016',
      '018',
      '019',
      '020',
      '021',
      '023',
      '025',
      '026']

#6. list of patients that are non-RS subtype
non_RS = ['003',
          '006',
          '009',
          '013',
          '022',
          '024']


#7. IPS PAtients STIM/OFF


ips_runs = {'01': ['000_DBS_OFF', '001_DBS_OFF', '003_DBS_OFF'],
          '02': ['006_C', '007_C', '008_C'],
          '03': ['000_A', '001_A', '002_A'],
          '04': ['006_C', '007_C', '008_C'],
          '05': ['000_A', '001_A', '002_A'],
          '06': ['003_B', '004_B', '005_B'],
          '07': ['007_C', '008_C', '009_C'],
          '08': ['000', '001', '002'],
          '09': ['006', '007', '008'],
          '10': ['000', '001', '002'],
          '11': ['003', '004', '005'],
          '12': ['006', '007', '008']}


side_preference = pd.DataFrame([{'subject': '001', 'ehi': 'rechts', 'flasche': 'rechts', 'triangle':'rechts'},
                                {'subject': '002', 'ehi': 'rechts', 'flasche': 'links ', 'triangle':'links'},
                                {'subject': '003', 'ehi': 'rechts', 'flasche': 'rechts', 'triangle':'rechts'},
                                {'subject': '006', 'ehi': 'rechts', 'flasche': 'links ', 'triangle':'links'},
                                {'subject': '007', 'ehi': 'rechts', 'flasche': 'rechts', 'triangle':'rechts'},
                                {'subject': '009', 'ehi': 'rechts', 'flasche': 'rechts', 'triangle':'rechts'},
                                {'subject': '010', 'ehi': 'rechts', 'flasche': 'links ', 'triangle':'rechts'},
                                {'subject': '012', 'ehi': 'rechts', 'flasche': 'links ', 'triangle':'links'},
                                {'subject': '013', 'ehi': 'rechts', 'flasche': 'links ', 'triangle':'links'},
                                {'subject': '014', 'ehi': 'rechts', 'flasche': 'links ', 'triangle':'links'},
                                {'subject': '016', 'ehi': 'rechts', 'flasche': 'rechts', 'triangle':'rechts'},
                                {'subject': '018', 'ehi': 'rechts', 'flasche': 'rechts', 'triangle':'rechts'},
                                {'subject': '019', 'ehi': 'rechts', 'flasche': 'rechts', 'triangle':'rechts'},
                                {'subject': '020', 'ehi': 'rechts', 'flasche': 'rechts', 'triangle':'rechts'},
                                {'subject': '021', 'ehi': 'rechts', 'flasche': 'links ', 'triangle':'links'},
                                {'subject': '022', 'ehi': 'rechts', 'flasche': 'rechts', 'triangle':'rechts'},
                                {'subject': '023', 'ehi': 'rechts', 'flasche': 'links ', 'triangle':'links'},
                                {'subject': '024', 'ehi': 'rechts', 'flasche': 'links ', 'triangle':'links'},
                                {'subject': '025', 'ehi': 'rechts', 'flasche': 'links ', 'triangle':'links'},
                                {'subject': '026', 'ehi': 'rechts', 'flasche': 'rechts', 'triangle':'rechts'}])



__version__ = 1.3
