import librosa


def load_raw(path, sr):
    x, sr = librosa.load(path) 
    return x
