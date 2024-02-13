self.gp.loc[:, 'distance'] =\
    np.sqrt(np.square(x_diff) + np.square(y_diff))
self.gp.loc[:, 'velocity'] = derivative(self.gp, 'distance')
self.gp.loc[:, 'acceleration'] = derivative(self.gp, 'velocity')
self.gp.loc[:, 'acceleration'] = (self.gp.loc[:, 'acceleration'] - self.gp.loc[:, 'acceleration'].mean()) / self.gp.loc[:, 'acceleration'].std()
