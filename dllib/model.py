from .imports import *
from .utils import *
from collections import OrderedDict

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

# From https://github.com/ncullen93/torchsample
def model_summary(m, input_size):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            #if is_listy(output):
            #    summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            #else:
            #    summary[m_key]['output_shape'] = list(output.size())
            #    summary[m_key]['output_shape'][0] = -1
            summary[m_key]['output_shape'] = list(output.size())
            summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and module.bias is not None:
                params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if (not isinstance(module, nn.Sequential) and
           not isinstance(module, nn.ModuleList) and
           not (module == m)):
            hooks.append(module.register_forward_hook(hook))

    summary = OrderedDict()
    hooks = []
    m.apply(register_hook)

    #if is_listy(input_size[0]):
    #    x = [to_gpu(Variable(torch.rand(3,*in_size))) for in_size in input_size]
    #else: x = [to_gpu(Variable(torch.rand(3,*input_size)))]
    x = [(Variable(torch.rand(3,*input_size)))]
    m(*x)

    for h in hooks: h.remove()
    return summary    