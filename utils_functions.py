import pickle
def get_model():
    model = pickle.load(open('model.sav', 'rb'))
    return model
x=get_model()