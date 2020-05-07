import numpy as np
import librosa
import os

train = 0
test = 0

samplerate = 22050
winlen = 512 # determined based on 512/22050 = 0.02321... ~ 23ms
shiftlen = 256
genre_dir = '.'
trainset = []
testset = []
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
rng = np.random.default_rng()
for root,_,files in os.walk(genre_dir):
	test_id_nr = rng.choice(100, 20, replace=False)
	print('New dir')
	for file in files:
		if file.endswith('.wav'):
			genre, idnr, _ = file.split('.')
			samples = librosa.load(os.path.join(root,file), sr=samplerate)[0]
			mfcc = librosa.feature.mfcc(samples, win_length=winlen, hop_length=shiftlen,n_mfcc=13)
			target = genres.index(genre)
			if int(idnr) in test_id_nr:
				# place in test set
				testset.append({'target':target, 'mfcc': mfcc})
			else:
				# place in training set
				trainset.append({'target':target, 'mfcc': mfcc})



np.savez_compressed('trainset.npz', trainset=trainset)
np.savez_compressed('testset.npz', testset=testset)