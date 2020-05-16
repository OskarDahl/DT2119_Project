import numpy as np
def load_data(file_path, one_hot=False):
    '''
    Input
        filepath: path to a .npz file
        one_hot: if true, targets are output as one-hot vectors
    Output
        mfccs: 13x190 mfcc maps
        targets: 0-9 genre target (or one-hot vectors)
    '''
    data = np.load(file_path, allow_pickle=True)
    key = list(data.keys())[0]
    data = data[key]
    mfccs = []
    targets = []
    ids = []
    for item in data:
        target = item['target']
        mfcc = item['mfcc']
        id = item['id']
        n_frames = mfcc.shape[1]
        idx = 0
        while(idx+190 < n_frames):
            targets.append(target)
            mfccs.append(mfcc[:,idx:idx+190])
            ids.append(id)
            idx += 190
    if one_hot:
        oh_targets = np.zeros((len(targets), max(targets)+1))
        oh_targets[np.arange(len(targets)),targets] = 1
        targets = oh_targets
    else:
        targets = np.array(targets)
    mfccs = np.stack(mfccs)
    return mfccs, targets, ids
