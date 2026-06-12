import numpy as np


def pointing_game(r, bbox):
    """
        Execute Pointing game with given relevance and bbox infomation 
        
        Args:
            r: Target relevance; (W, H)
            bbox: Ground-truth information for bbox in the image; (x, y, width, hieght)
        
        Returns:
            r_along_e: List of masked relevances per energes
            scores: List of pointing game scores per energes 
    """
    
    # energe_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01]
    energe_list = list(np.arange(0, 1, 0.05)+0.05)
    rw, rh = r.shape
    nb_r = len(np.where(r>0)[0])
    # print('nb_r: ', nb_r)
    
    # row_min, col_min, row_max, col_max
    pltpoint_to_rowcol = [bbox[1], bbox[0], bbox[1]+bbox[3], bbox[0]+bbox[2]]
    # print('pltpoint_to_rowcol: ', pltpoint_to_rowcol)
    
    # 나중에 확장 필요 
    r_along_e = []
    # nb_r_along_e = []
    scores = {}
    for e in energe_list:
        q = np.quantile(r, (1-e))
        m = r>=q
        r_e = r*m
        nb_re = len(np.where(r_e>0)[0])
        
        # r_along_e.append(r_e)
        # nb_r_along_e.append(nb_re)
        
        hit = 0
        miss = 0
        for i in range(rw):
            for j in range(rh):
                if r_e[i,j] > 0:
                    if i>=pltpoint_to_rowcol[0] and i<pltpoint_to_rowcol[2] and \
                        j>=pltpoint_to_rowcol[1] and j<pltpoint_to_rowcol[3]: hit += 1
                    else: miss += 1
                        
        scores[e] = (hit/(hit+miss))
    
    return scores