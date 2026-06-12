import numpy as np
import torch


def LeRF_perturbation(r, x, y, model, device, nb_iter=100, margin=200):
    scores = {}
    
    b, w, h = r.shape
    r = r.reshape(b, w*h)
    r_idx_asc = np.argsort(r, axis=1)
    
    s_idx = 0
    e_idx = margin
    
    for i in range(1, nb_iter+1):
        nb_pb_pixels = i*margin
        
        r_idx = r_idx_asc[:, s_idx:e_idx]
        r_idx = np.transpose(r_idx)
        
        m = torch.ones(x.shape, dtype=torch.float32).to(device)
        
        for idx_row in r_idx:
            for j, idx_row_each in enumerate(idx_row):
                rx, ry = idx_row_each//w, idx_row_each%h
                m[j, :, rx, ry] = 0
        
        x *= m
        
        s_idx = e_idx
        e_idx += margin
        
        output = model(x)
        _, p =  torch.max(output.data, 1)
        scores[nb_pb_pixels] = (p == y).sum().item()
        
    return scores