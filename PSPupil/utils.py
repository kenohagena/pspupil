import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyxdf

'''
Utility functions, e.g. for loading specific data formats
'''



def load_xdf(path, pupil_stream = 1):
    '''
    Load xdf.files that contain pupils
    Input:
    - path to xdf-file
    - position of pupil-stream within xdf file
    Output:
    - gaze pd.DataFrame
    - left pupil diameter pd.DataFrame
    - right pupil diameter pd.DataFrame
    '''
    streams, header = pyxdf.load_xdf(path)
    try:
        pupil_stream = streams[1]
        cols = [i['label'][0] for i in pupil_stream['info']['desc'][0]['channels'][0]['channel']]
        cols = np.array(cols)
    except TypeError:
        pupil_stream = streams[0]
        cols = [i['label'][0] for i in pupil_stream['info']['desc'][0]['channels'][0]['channel']]
        cols = np.array(cols)
    if 'diameter1_2d' not in cols:
        pupil_stream = streams[0]
        cols = [i['label'][0] for i in pupil_stream['info']['desc'][0]['channels'][0]['channel']]
        cols = np.array(cols)

    df = pd.DataFrame(pupil_stream['time_series'])
    df.columns = cols
    time = pd.DataFrame(pupil_stream['time_stamps'])
    df['time'] = time.iloc[:, 0].values
    df.time = df.time - df.time.values[0]

    left = df.loc[:, ['diameter1_2d', 'time', 'confidence']].\
    drop_duplicates(subset='diameter1_2d').rename(columns = {'diameter1_2d':'diameter',
                                                            'time': 'timestamp'})

    right = df.loc[:, ['diameter0_2d', 'time', 'confidence']].\
    drop_duplicates(subset='diameter0_2d').rename(columns = {'diameter0_2d':'diameter',
                                                            'time': 'timestamp'})

    gaze = df.loc[:, ['norm_pos_x', 'norm_pos_y', 'time', 'confidence']].rename(columns = {'diameter0_2d':'diameter',
                                                                                            'time': 'timestamp',
                                                                                           'norm_pos_x': 'x',
                                                                                           'norm_pos_y': 'y'})
    return left, right, gaze


def plot_single_trace(pupilframe, sub = 'Subject', session = 'Session', run = 'Run',
    confidence_thresh=.9, excel_thresh=2):
    f, ax = plt.subplots(figsize=(40, 5))
    mean= pupilframe.diameter_mean.mean() -5
    ax.plot((pupilframe.diameter_mean.values ), color='black', alpha=.5)
    ax.plot((pupilframe.d_intp.values), color='red', alpha=.5)

    ax.plot(pupilframe.confidence_d.values*100, color='green', alpha=.25)
    ax.axhline(confidence_thresh*100, color='red')
    ax.plot(pupilframe.acceleration.values*5, color='blue', alpha=.25)
    ax.axhline(excel_thresh*5, color='red')
    ax.set_ylim(0, 100)
    f.suptitle('{0} - {1} - {2}'.format(sub, session, run))



__version__ = 1.1
