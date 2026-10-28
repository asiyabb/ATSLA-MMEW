import collections
import torch

def load_pretrained_weights(model, checkpoint):
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers = []
    
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
            
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print(f"Successfully loaded {len(matched_layers)} pre-trained layers.")
    return model