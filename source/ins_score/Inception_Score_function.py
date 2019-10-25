from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn.functional as F


def Inception_Score(images_set, model_predictor,
                    device=torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu"), n=10, bs=100):
    """
    Calculates Inception score for images, created with some generator

    Parameters
    ----------
    images_set : Torch Dataset
        Dataset that returns generated image
    model_predictor : PyTorch model
        Model that extract feature vector of the image for calculating KL divergence
    n : int
        Number of splits of images set
    bs : int
        Batch size
    Returns
    -------
    IS : float
        Mean of IS score
    IS_err : float
        Error of IS score 
    """
    from math import floor
    model_predictor.eval()

    test_loader = DataLoader(images_set, batch_size=bs, shuffle=False)

    predictions = []
    scores = []
    for i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            logits = model_predictor(data)
            p_yx = F.softmax(logits, dim=1).cpu().numpy()
        predictions.append(p_yx)

    size = n * (len(predictions) // n)
    predictions = np.vstack(predictions[:size])
    predictions = predictions.reshape(n, len(predictions) // n, 1000)
    for i in range(n):
        p_yx = predictions[i]
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        KL = ((p_yx * (np.log(p_yx + 1e-6) - np.log(p_y + 1e-6))).sum(axis=1)).mean()
        IS = np.exp(KL)
        scores.append(IS)

    IS = np.mean(scores)
    IS_err = np.std(scores)
    return IS, IS_err
