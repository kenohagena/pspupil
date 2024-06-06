import numpy as np
import pandas as pd
import pickle
from glob import glob
from os.path import join
from scipy import signal
from scipy import interpolate
from decim.adjuvant import slurm_submit as slu
from datetime import datetime
import matplotlib.pyplot as plt
from memoization import cached



def island(array, threshhold=200):
    '''
    INPUT: np.array with valid samples as TRUE, invalid as FALSE
    OPERATION: Detects streaks of TRUE samples smaller than threshhold
    ARGUMENTS:
        - threshhold: length of maximum streak length

    OUTPUT: np.array as input, valid=TRUE, invalid=FALSE
    '''
    convolved = np.convolve(~array, [0.5, 1], 'same')
    ev_start = np.where(convolved == .5)[0]
    fragment_ends = ev_start
    if convolved[0] != 0:
        fragment_ends = fragment_ends[1:len(fragment_ends)]
    if convolved[len(convolved) - 1] == 0:
        fragment_ends = np.append(fragment_ends, len(array))
    ev_end = np.where(convolved == 1)[0]
    if convolved[0] == 0:
        fragment_starts = np.append(0, ev_end)
    else:
        fragment_starts = ev_end
    if convolved[-1] == 1:
        fragment_starts = fragment_starts[0:-1]
    assert len(fragment_ends) == len(fragment_starts)
    fragment_length = fragment_ends - fragment_starts
    wh = np.where(fragment_length < threshhold)
    smallfrag_ends = fragment_ends[wh]
    smallfrag_starts = fragment_starts[wh]
    for start, end in zip(smallfrag_starts, smallfrag_ends):
        array[start:end + 1] = False
    return array


def margin(array, margin1=50, margin2=50, threshhold=50, op='larger'):
    '''
    Detect fragments of invalid samples of certain length,
    mark surrounding samples as invalid
    INPUT: boolean np.array; valid samples True & invalid samples FALSE
    ARGUMENTS:
        - margin 1: margin before fragment
        - margin 2: margin after fragment
        - threshhold: length of fragment of invalid samples
        - op: 'larger' or 'smaller'; how to use threshhold
    OUTPUT: boolean np.array; valid samples True & invalid samples FALSE
    '''
    convolved = np.convolve(array, [0.5, 1], 'same')                    # convolution marks begin of True-series with 0.5 and end of True-series with 1
    ev_start = np.where(convolved == .5)[0]
    fragment_ends = ev_start
    if convolved[0] != 0:
        fragment_ends = fragment_ends[1:len(fragment_ends)]
    if convolved[len(convolved) - 1] == 0:
        fragment_ends = np.append(fragment_ends, len(array))
    ev_end = np.where(convolved == 1)[0]
    if convolved[0] == 0:
        fragment_starts = np.append(0, ev_end)
    else:
        fragment_starts = ev_end
    if convolved[-1] == 1:
        fragment_starts = fragment_starts[0:-1]
    assert len(fragment_ends) == len(fragment_starts)
    fragment_length = fragment_ends - fragment_starts
    if op == 'larger':
        wh = np.where(fragment_length > threshhold)
    elif op == 'smaller':
        wh = np.where(fragment_length <= threshhold)
    smallfrag_ends = fragment_ends[wh]
    smallfrag_starts = fragment_starts[wh]
    for start, end in zip(smallfrag_starts, smallfrag_ends):
        array[start - margin1:start] = False
        array[end:end + margin2] = False
    return array


def derivative(df, unit):
    '''
    Assumes that entries along unit axis are evenly spaced in time
    If not, needs to be divided by temporal offset between points
    '''
    f = np.abs((df[unit].values - np.roll(df[unit].values, -1)))
    b = np.abs((df[unit].values - np.roll(df[unit].values, 1)))
    return np.maximum(f, b)


class PupilFrame(object):

    def __init__(self, subject, group, session, run,
                 directory='/Volumes/psp_data/', out_dir='', version = 'old'):
        '''
        Initialize
        --------------------------------------------------------------
        ARGUMENTS:
        - subject: e.g. '001'
        - group: 'patient'
        - session: 'Baseline' / 'Followup'
        - run: '000', etc.
        - directory: e.g. '/Volumes/XKCD/PSP'
        '''
        self.subject = subject
        self.group = group
        self.session = session
        self.run = run
        self.directory = directory
        self.pupil = {}
        self.parameters = {}
        self.version = version
        if out_dir == '':
            self.out_dir = join(self.directory, 'Pupil_Preprocessed_{}'.
                                format(datetime.now().strftime("%Y-%m-%d")), self.group)
        else:
            self.out_dir = join(out_dir, 'Pupil_Preprocessed_{}'.
                                format(datetime.now().strftime("%Y-%m-%d")), self.group)
        slu.mkdir_p(self.out_dir)

    def load_pupil(self):
        '''
        Load pupil data and discard irrelevant data.
        '''
        if self.group == 'PSP':
            path = glob(join(self.directory, 'data', 'raw_renamed', 'PSP',
                                        '20*_PSP_{0}_{1}'.format(self.subject,
                                                      self.session),
                             self.run, 'pupil_data'))
        elif self.group == 'Control':
            path = glob(join(self.directory, 'data', 'raw_renamed', 'CONTROLS',
                             '*K{}'.format(self.subject),
                             self.run, 'pupil_data'))

        elif self.group == 'PD':
            path = glob(join(self.directory, 'data', 'raw_renamed', 'PD',
                             '2017*P{}'.format(self.subject),
                             self.run, 'pupil_data'))

        with open(path[0], 'rb') as f:
            file = pickle.load(f, encoding="latin-1")
        df = pd.DataFrame(file['pupil_positions'])
        self.pupil['left'] = df.loc[df.id == 0].drop(['norm_pos',
                                                      'ellipse',
                                                      'method'], axis=1)
        self.pupil['right'] = df.loc[df.id == 1].drop(['norm_pos',
                                                       'ellipse',
                                                       'method'], axis=1)
        self.pupil['gaze'] = pd.DataFrame(file['gaze_positions'])

    def load_csv(self, path):
        df = pd.DataFrame(pd.read_csv(path))
        self.pupil['left'] = df.loc[df.id == 0].drop(['norm_pos',
                                                      'ellipse',
                                                      'method'], axis=1)
        self.pupil['right'] = df.loc[df.id == 1].drop(['norm_pos',
                                                       'ellipse',
                                                       'method'], axis=1)
        self.pupil['gaze'] = pd.DataFrame(file['gaze_positions'])


    def cut_resample(self, start=5, end=210):
        '''
        INPUT: pupil diameter, gaze dataframe
        OPERATION:
            - Cut all data at 5:210
            - resample/interpolate to evenly spaced frequency to align
            pupil left, right, gaze
        ARGUMENTS:
            - start and end for cut
        OUTPUT:
            - combined pupil dataframe with timestamps as index
        '''
        time_zero = np.min([self.pupil['left'].iloc[0].timestamp,
                            self.pupil['right'].iloc[0].timestamp,
                            self.pupil['gaze'].iloc[0].timestamp])                    # reference timepoint
        for side in ['left', 'right', 'gaze']:
            df = self.pupil[side]
            if side is 'gaze':
                if self.version == 'old':
                    df[['x', 'y']] = pd.DataFrame(df['norm_pos'].tolist(),
                                                  index=df.index)

            df.loc[:, 'time'] =\
                np.round((df.timestamp - time_zero) * 1000)

            df = df.set_index(pd.to_datetime(df['time'], unit='ms'))
            df = df.resample('ms').mean().interpolate('linear')          # upsample to 'ms'
            df = df.resample('17ms').mean()                              # downsample to 60Hz
            df = df.loc[pd.Timestamp(start, unit='s'):pd.Timestamp(end, unit='s')]
            new_cols = {'diameter': 'diameter_{}'.format(side),
                        'confidence': 'confidence_{}'.format(side),
                        'norm_pos': 'norm_pos'}

            df = df.rename(columns=new_cols)
            self.pupil[side] = df
        self.gp = pd.concat([self.pupil['left'], self.pupil['right'],
                             self.pupil['gaze']], axis=1)

    def excel(self):
        '''
        INPUT: df with gaze coordinate vectors as columns 'x' and 'y'
        OPERATION: compute distance in euclidian space, velocity and acceleration
        '''

        x_diff = self.gp.x.values - np.roll(self.gp.x.values, 1)
        x_diff[0] = 0

        y_diff = self.gp.y.values - np.roll(self.gp.y.values, 1)
        y_diff[0] = 0
        self.gp.loc[:, 'distance'] =\
            np.sqrt(np.square(x_diff) + np.square(y_diff))
        self.gp.loc[:, 'velocity'] = derivative(self.gp, 'distance')
        self.gp.loc[:, 'acceleration'] = derivative(self.gp, 'velocity')
        self.gp.loc[:, 'acceleration'] = (self.gp.loc[:, 'acceleration'] - self.gp.loc[:, 'acceleration'].mean()) / self.gp.loc[:, 'acceleration'].std()

    def discard_interp(self, excel_thresh, confidence_thresh,
                       islands, margin1, margin2):
        '''
        INPUT: pd.DataFrame with both eyes diameter,
        gaze-acceleration and confidence estimates
        OPERATION:
            - discard sampples based on i) confidence of both eyes and
                                        ii) gaze acceleration
                                        iii) small 'islands'
            - interpolate linearly using pd.interpolate
        ARGUMENTS:
            - acceleration threshhold
            - confidence threshshold
            - island threshhold
        '''

        self.gp.loc[:, 'notes'] = np.nan
        # take better tracked pupil based on confidence
        if (self.gp.confidence_right < confidence_thresh).mean() >\
                (self.gp.confidence_left < confidence_thresh).mean():
            self.gp.loc[:, 'diameter_mean'] = self.gp.loc[:, 'diameter_left']
            self.gp.loc[:, 'confidence_d'] = self.gp.loc[:, 'confidence_left']
        elif (self.gp.confidence_left < confidence_thresh).mean() >\
                (self.gp.confidence_right < confidence_thresh).mean():
            self.gp.loc[:, 'diameter_mean'] = self.gp.loc[:, 'diameter_right']
            self.gp.loc[:, 'confidence_d'] = self.gp.loc[:, 'confidence_right']
        else:
            print("Confidence equal in both eyes")
            self.gp.loc[:, 'diameter_mean'] = self.gp.loc[:, 'diameter_left']
            self.gp.loc[:, 'confidence_d'] = self.gp.loc[:, 'confidence_left']

        self.gp.loc[:, 'diameter_blink'] = self.gp.loc[:, 'diameter_mean'].copy()
        self.gp.loc[self.gp.acceleration.abs() > excel_thresh, 'diameter_blink'] = np.nan
        self.gp.loc[self.gp.confidence_d < confidence_thresh, 'diameter_blink'] = np.nan
        self.gp.loc[:, 'margin'] = self.gp.diameter_blink
        self.gp.loc[:, 'margin'] = ~self.gp.margin.isnull()
        self.gp.loc[:, 'margin'] = margin(self.gp.margin.values, margin1=margin1, margin2=margin2, threshhold=1)
        self.gp.loc[self.gp.margin == False, 'diameter_blink'] = np.nan

        array = ~self.gp.diameter_blink.isnull()
        array = island(array, threshhold=islands)
        self.gp.loc[:, 'islands'] = array
        self.gp.loc[self.gp.islands == False, 'diameter_blink'] = np.nan
        self.gp.loc[:, 'd_intp'] =\
            self.gp.diameter_blink.interpolate('linear')

    def discard_interp_(self, excel_thresh, confidence_thresh,
                       islands, margin1, margin2):
        '''
        second function if better pupil already chosen
        INPUT: pd.DataFrame with both eyes diameter,
        gaze-acceleration and confidence estimates
        OPERATION:
            - discard sampples based on i) confidence of both eyes and
                                        ii) gaze acceleration
                                        iii) small 'islands'
            - interpolate linearly using pd.interpolate
        ARGUMENTS:
            - acceleration threshhold
            - confidence threshshold
            - island threshhold
        '''

        self.gp.loc[:, 'diameter_blink'] = self.gp.loc[:, 'diameter_mean'].copy()
        self.gp.loc[self.gp.acceleration.abs() > excel_thresh, 'diameter_blink'] = np.nan
        self.gp.loc[self.gp.confidence_d < confidence_thresh, 'diameter_blink'] = np.nan
        self.gp.loc[:, 'margin'] = self.gp.diameter_blink
        self.gp.loc[:, 'margin'] = ~self.gp.margin.isnull()
        self.gp.loc[:, 'margin'] = margin(self.gp.margin.values, margin1=margin1, margin2=margin2, threshhold=1)
        self.gp.loc[self.gp.margin == False, 'diameter_blink'] = np.nan

        array = ~self.gp.diameter_blink.isnull()
        array = island(array, threshhold=islands)
        self.gp.loc[:, 'islands'] = array
        self.gp.loc[self.gp.islands == False, 'diameter_blink'] = np.nan
        self.gp.loc[:, 'd_intp'] =\
            self.gp.diameter_blink.interpolate('linear')

    def normalize(self):
        '''
        z-score
        '''
        sig = self.gp.loc[:, 'd_intp']
        self.gp.loc[:, 'biz'] = (sig - sig.mean()) / sig.std()

    def save(self):
        self.gp = self.gp.drop(['id', 'time', 'timestamp'], axis=1, errors='ignore')  # to_hdf does not work if columsn names are double

        self.gp.to_hdf(join(self.out_dir, 'Pupil_Preprocessed_SUB-{0}_{1}_{2}.hdf'.
                            format(self.subject,
                                   self.session,
                                   self.run)), key='Pupil')

@cached
def execute(subject, group, session, run, directory='/Volumes/XKCD/PSP',
            start=5, end=210, excel_thresh=0.05, confidence_thresh=.9,
            islands=10, margin1=10, margin2=10, out_dir=''):
    p = PupilFrame(subject, group, session, run, directory, out_dir=out_dir)
    try:
        p.load_pupil()
        p.cut_resample(start=start, end=end)
        p.excel()
        p.discard_interp(excel_thresh=excel_thresh, confidence_thresh=confidence_thresh,
                         islands=islands, margin1=margin1, margin2=margin2)
        p.normalize()
        p.save()
        return p
    except IndexError:
        print('File Error for', subject, session, run)  # glob finds no files


def adjust(subject, group, session, run, directory='/Volumes/XKCD/PSP',
           start=5, end=210, excel_thresh=0.05, confidence_thresh=.9,
           islands=10, margin1=10, margin2=10, out_dir='', parameters=False):
    p = PupilFrame(subject, group, session, run, directory, out_dir=out_dir)
    p.load_pupil()
    p.cut_resample(start=start, end=end)
    p.excel()
    p.discard_interp(excel_thresh=excel_thresh, confidence_thresh=confidence_thresh,
                     islands=islands, margin2=margin2, margin1=margin1)
    f, ax = plt.subplots(figsize=(40, 5))
    ax.plot(p.gp.diameter_mean.values, color='black', alpha=.5)
    ax.plot(p.gp.d_intp.values, color='red', alpha=.5)

    if parameters is True:
        m = p.gp.diameter_mean.max()
        ax.plot(p.gp.confidence_d.values*m*1.8, color='green', alpha=.25)
        ax.axhline(confidence_thresh*m*1.8, color='red')
        ax.plot(p.gp.acceleration.values*m/15, color='blue', alpha=.25)
        ax.axhline(excel_thresh*m/15, color='red')
        ax.set_ylim(0, m*1.8)

    return p



__Version__ = 1.15
