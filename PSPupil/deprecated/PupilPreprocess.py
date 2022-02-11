import numpy as np
import pandas as pd
import pickle
from glob import glob
from os.path import join
from scipy import signal
from scipy import interpolate
from decim.adjuvant import slurm_submit as slu
from datetime import datetime


def fill_nan(A, kind='linear'):
    '''
    Interpolate to fill nan values
    INPUT: np.array of data series to interpolate np.nan values
    ARGUMENTS:
        - kind= 'linear' oder 'cubic'
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good], bounds_error=False,
                             kind=kind)
    B = np.where(np.isfinite(A), A, f(inds))
    return B


def interpol(pupil, nan_locs=False, valid=[], kind='linear'):
    '''
    Do interpolation of np.nan values; extrapolate at start and end
    INPUT: Pupil array to be interpolated as np.array astype float
    ARGUMENTS:
        - nan_locs: True when array with valid samples is given,
        otherwise just interpolate the pupil array
        - valid: np.array of same length as pupil;
                    invalid samples FALSE & valid samples TRUE
        - kind: 'linear' or 'cubic' interpolation
    OUTPUT: array with interpolated pupil data
    '''
    if nan_locs is True:
        pupil[np.where(~valid)] = np.nan
    interp = fill_nan(pupil, kind=kind)
    interp = pd.DataFrame(interp).fillna(method='ffill')[0].values      # NaN at the end ffilled
    interp = pd.DataFrame(interp).fillna(method='bfill')[0].values      # NaN at the beginning bfilled
    return interp

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


def islands(array, threshhold=200):
    '''
    Detects leftover fragments smaller than threshhold.

    Sets those detected fragments to NaN to make linear interpolation cleaner.
    '''
    convolved = np.convolve(array, [0.5, 1], 'same')
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


def velocity(df, unit, time):
    f = np.abs((df[unit].values - np.roll(df[unit].values, -1)) /
               (df[time].values - np.roll(df[time].values, -1)))
    b = np.abs((df[unit].values - np.roll(df[unit].values, 1)) /
               (df[time].values - np.roll(df[time].values, 1)))
    return np.maximum(f, b)

def lowpass(array, cutoff=4, freq=125):

    lp_cof_sample = cutoff / (freq / 2)                                 # low pass
    blp, alp = signal.butter(3, lp_cof_sample)
    filtered = signal.filtfilt(blp, alp, array)
    return filtered


class PupilFrame(object):

    def __init__(self, subject, group, session, run,
                 directory='/Volumes/XKCD/PSP', out_dir=''):
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
        if out_dir == '':
            self.out_dir = join(self.directory, 'Pupil_Preprocessed_{}'.
                                format(datetime.now().strftime("%Y-%m-%d")))
        else:
            self.out_dir = join(out_dir, 'Pupil_Preprocessed_{}'.
                                format(datetime.now().strftime("%Y-%m-%d")))
        slu.mkdir_p(self.out_dir)

    def load_pupil(self):
        '''
        Load pupil data and discard irrelevant data.
        '''
        path = glob(join(self.directory, 'raw_data', self.subject,
                         '20*_PSP_{0}_{1}'.format(self.subject,
                                                  self.session),
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

    def cut(self, side, n=3):
        '''
        Discard frames at begin and end of pupil data
        ARGUMENTS:
            - pupil side
            - length of margin in seconds
        '''
        self.pupil[side]
        m = n * 125
        self.pupil[side] = self.pupil[side].iloc[m:-m]

    def discard_basic(self, side, conf_cutoff=.9, d_cutoff=.98):
        '''

        '''
        df = self.pupil[side]
        df.loc[:, 'diameter_conf'] = df.diameter.copy()
        df.loc[df.confidence < conf_cutoff, 'diameter_conf'] = np.nan
        df.loc[df.diameter.abs() > df.diameter.abs().quantile(d_cutoff),
               'diameter_conf'] = np.nan
        self.pupil[side] = df

    def discard_velocity(self, side, n=1):
        df = self.pupil[side]
        df.loc[:, 'timestamp'] = \
            (df.timestamp - df.timestamp.values[0]) * 1000
        df.loc[:, 'dilation_speed'] = velocity(df, unit='diameter',
                                               time='timestamp')
        velo = df.dilation_speed.values
        median = df.dilation_speed.quantile(.5)
        MAD = np.median(np.abs(velo - median))
        threshhold = median + n * MAD
        df.loc[:, 'diameter_velo'] = df['diameter_conf'].copy()
        df.loc[df.dilation_speed > threshhold, 'diameter_velo'] = np.nan
        self.pupil[side] = df

    def interp1(self, side, sample_rate=125, margins=[7, 50],
                margin_threshhold=20, island_threshhold=25, lpf=4):
        df = self.pupil[side]
        df.loc[:, 'margins'] = ~df.diameter_velo.isnull()
        print(df.margins.mean())
        df.loc[:, 'margins'] = margin(df.loc[:, 'margins'].values,
                                      margin1=margins[0],
                                      margin2=margins[1],
                                      threshhold=margin_threshhold)
        print(df.margins.mean())
        df.loc[:, 'margins'] = islands(df.loc[:, 'margins'].values,
                                       threshhold=island_threshhold)
        print(df.margins.mean())
        df.loc[:, 'interpolated'] = interpol(pupil=df.diameter.copy().values,
                                             nan_locs=True,
                                             valid=df.margins.values,
                                             kind='linear')

        df.loc[:, 'filtered'] = lowpass(df['interpolated'], cutoff=lpf)
        self.pupil[side] = df

    def repeat(self, side, i, n=2.5, margins=[7, 50],
               margin_threshhold=20, island_threshhold=2, lpf=4):
        df = self.pupil[side]
        if i == 0:
            filtered = df['filtered']
        else:
            filtered = df['filtered_{}'.format(i - 1)]

        trend_diff = df['diameter_velo'] - filtered
        median = trend_diff.quantile(.5)
        MAD = np.median((trend_diff - median).abs().quantile(.5))
        threshhold = median + n * MAD
        df.loc[:, 't'] = trend_diff
        df.loc[:, 'td'] = df['diameter_velo'].copy()
        df.loc[df.t.abs() > threshhold, 'td'] = np.nan
        trend_line_dev = df['td'].values
        marg = ~np.isnan(trend_line_dev)
        marg = margin(marg, margin1=margins[0], margin2=margins[1],
                      threshhold=margin_threshhold)
        marg = islands(marg, threshhold=island_threshhold)
        print(1 - df['diameter_velo'].isnull().mean(),
              np.mean(~np.isnan(trend_line_dev)),
              np.mean(marg))
        df.loc[:, 'interpolated_{}'.format(i)] =\
            interpol(pupil=df.diameter.copy().values,
                     nan_locs=True, valid=marg,
                     kind='linear')
        df.loc[:, 'filtered_{}'.format(i)] = \
            lowpass(df['interpolated_{}'.format(i)].values, cutoff=lpf)
        self.pupil[side] = df

    def highpass(self, side, highpass=.01, sample_rate=125):
        '''
        Apply 3rd-order Butterworth highpass filter.
        '''

        sig = self.pupil[side].loc[:, 'filtered_19']                        # High pass:
        hp_cof_sample = highpass / (sample_rate / 2)
        bhp, ahp = signal.butter(3, hp_cof_sample, btype='high')
        highpassed = signal.filtfilt(bhp, ahp, sig)
        self.pupil[side]['highpass'] = highpassed

    def normalize(self, side):
        '''
        z-score
        '''
        sig = self.pupil[side].loc[:, 'highpass']
        self.pupil[side]['biz'] = (sig - sig.mean()) / sig.std()

    def save(self):

        for side in ['left', 'right']:
            self.pupil[side].\
                to_hdf(join(self.out_dir, 'Pupil_Preprocessed_SUB-{0}_{1}_{2}.hdf'.
                            format(self.subject,
                                   self.session,
                                   self.run)), key=side)


def execute(subject, group, session, run, directory='/Volumes/XKCD/PSP',
            velocity_threshhold=1, trendline_threshhold=2.5, lpf=4,
            margins=[7, 50], margin_threshhold=20, island_threshhold=25,
            cut_frames=3, conf_cutoff=.9, d_cutoff=.98, ret=False, out_dir=''):
    p = PupilFrame(subject, group, session, run, directory, out_dir=out_dir)
    try:
        p.load_pupil()
        for side in ['left', 'right']:
            p.cut(side=side, n=cut_frames)
            p.discard_basic(side=side, conf_cutoff=conf_cutoff,
                            d_cutoff=d_cutoff)
            p.discard_velocity(side=side, n=velocity_threshhold)
            p.interp1(side=side, margins=margins,
                      margin_threshhold=margin_threshhold,
                      island_threshhold=island_threshhold, lpf=lpf)
            for i in range(20):
                p.repeat(i=i, side=side, n=trendline_threshhold,
                         margins=margins,
                         margin_threshhold=margin_threshhold,
                         island_threshhold=island_threshhold, lpf=lpf)
            p.highpass(side=side)
            p.normalize(side=side)
        p.save()
        if ret is True:
            return(p.pupil)
    except IndexError:
        print('File Error for', subject, session, run)  # glob finds no files


'''
Test code:

subject = '001'
group = 'patient'
session = 'Baseline'
run = '000'
execute(subject, group, session, run)


V.1.0.1
Second pupil preprocessing script for PSP dataset
'''
