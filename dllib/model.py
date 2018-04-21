from .imports import *
from .utils import *

def get_prediction(x):
    if is_listy(x): x=x[0]
    return x.data

def predict(m, dl):
    preda,_ = predict_with_targs_(m, dl)
    return to_np(torch.cat(preda))

def predict_batch(m, x):
    m.eval()
    #if hasattr(m, 'reset'): m.reset()
    return m(make_var(x))

def predict_with_targs_(m, dl):
    m.eval()
    #if hasattr(m, 'reset'): m.reset()
    res = []
    for x,y in tqdm(iter(dl)): res.append([get_prediction(F.softmax(m(make_var(x)),dim=1)),y])
    return zip(*res)

def predict_with_targs(m, dl):
    preda,targa = predict_with_targs_(m, dl)
    return to_np(torch.cat(preda)), to_np(torch.cat(targa))
    


