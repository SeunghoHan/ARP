import numpy as np

def pointing_game(r, mask_in, mask_out):
    # energe_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01]
    energe_list = list(np.arange(0, 1, 0.05)+0.05)
    rw, rh = r.shape[1:]
    nb_r = len(np.where(r>0)[0])
    
    scores = {}
    for e in energe_list:
        scores[e] = 0
    
    for e in energe_list:
        q = np.quantile(r, (1-e), axis=[1,2])
        for i, q_i in enumerate(q):
            r_i = r[i]
            m_i = r_i >= q_i
            r_i_e = r_i * m_i
        
            mask_in_i = mask_in[i]
            mask_out_i = mask_out[i]
            
            r_mask_in = r_i_e * mask_in_i
            r_mask_out = r_i_e * mask_out_i
            
            hit = len(np.where(r_mask_in > 0)[0])
            miss = len(np.where(r_mask_out > 0)[0])
            
            if hit+miss == 0:
                scores[e] += 0
            else:
                scores[e] += (hit/(hit+miss))
    
    return scores