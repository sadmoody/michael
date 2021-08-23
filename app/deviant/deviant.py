import pickle
from .feature_extractor import FeatureExtractor

fe = FeatureExtractor()
scaler = pickle.load(open("app/deviant/pickle/scaler.pickle", 'rb'))
BOTTOM = 1
TOP = 16
svm = []
for i in range(BOTTOM, TOP+1):
    svm.append(pickle.load(open(f"app/deviant/pickle/svm{i}.pickle", 'rb')))

def infer(sent, sensitivity=8.0):
    sensitivity = round(sensitivity)
    sensitivity = max(BOTTOM, sensitivity)
    sensitivity = min(TOP, sensitivity)
    return svm[sensitivity-1].predict(scaler.transform(fe.process([sent])))[0] == 1