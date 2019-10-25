from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn.functional as F
from scipy.linalg import sqrtm

def Frеchet_Inception_Distance(orig_images, gen_images, model_predictor, device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu"),bs = 100):
    """
    Calculates FID beеween original and generated images

    Parameters
    ----------
    orig_images : Torch Dataset
        Dataset that returns generated image
    gen_images : Torch Dataset
        Dataset that returns generated image
    model_predictor : PyTorch model
        Model that extract feature vector of the image for calculating KL divergence
    bs : int
        Batch size
    Returns
    -------
    FID : float
    """
    model_predictor.eval()
    
    orig_loader = DataLoader(orig_images,batch_size = bs,shuffle=False)
    
    orig_predictions = []
    for i,data in enumerate(orig_loader):
        data=data.to(device)
        with torch.no_grad():
            logits = model_predictor(data)
            p_yx = F.softmax(logits, dim = 1).cpu().numpy()
        orig_predictions.append(p_yx)
    
    orig_predictions = np.vstack(orig_predictions)
    
    gen_loader = DataLoader(gen_images,batch_size = bs,shuffle=False)
    
    gen_predictions = []
    for i,data in enumerate(gen_loader):
        data=data.to(device)
        with torch.no_grad():
            logits = model_predictor(data)
            p_yx = F.softmax(logits, dim = 1).cpu().numpy()
        gen_predictions.append(p_yx)
    
    gen_predictions = np.vstack(gen_predictions)
    
    orig_mean = orig_predictions.mean(axis = 0)
    gen_mean = gen_predictions.mean(axis = 0)
    
    
    
    orig_cov = np.cov(orig_predictions,rowvar=False)
    gen_cov = np.cov(gen_predictions,rowvar=False)
    
    mean_dist = ((orig_mean - gen_mean)**2).sum()
    cov_dist = np.trace(orig_cov+ gen_cov-2*sqrtm(orig_cov@gen_cov))
    if np.iscomplexobj(cov_dist):
        cov_dist = cov_dist.real
        
    
    return mean_dist+cov_dist