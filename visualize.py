from math import ceil

import torch
import numpy as np
import cv2


def mark_image(im):
    im[:10, :, 0] = 0
    im[:10, :, 1] = 255
    im[:10, :, 2] = 0
    im[-10:, :, 0] = 0
    im[-10:, :, 1] = 255
    im[-10:, :, 2] = 0
    im[:, :10, 0] = 0
    im[:, :10, 1] = 255
    im[:, :10, 2] = 0
    im[:, -10:, 0] = 0
    im[:, -10:, 1] = 255
    im[:, -10:, 2] = 0
    return im


def create_viz(im, im_wo, scores, masks, activity_classes):
    if im_wo is None:
        im = np.hstack((im, im))
    else:
        im = np.hstack((im, im_wo))
    
    num_classes = masks.size()[1]
    scores = scores.view(-1, num_classes)
    pred = scores.data.max(1)[1].item()

    mask_shape = (ceil(im.shape[1] / num_classes), ceil(im.shape[1] / num_classes))
    hms = []
    for cl in range(num_classes):
        hm = 255*cv2.resize(masks[0, cl, :, :].view(15, 15).detach().cpu().numpy(), mask_shape)
        hm = cv2.applyColorMap(hm.astype(np.uint8), cv2.COLORMAP_JET)
        if pred == cl:
            hm = mark_image(hm)
        hms.append(hm)

    hms = np.hstack(hms)
    excess = hms.shape[1] - im.shape[1]
    im_out = np.vstack((im, hms[:, :-excess, :]))
    im_out = im_out.astype(np.uint8)

    cv2.putText(im_out, 'Original Image', (int(im_out.shape[1] / 4 - 50), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5)
    cv2.putText(im_out, 'Original Image', (int(im_out.shape[1] / 4 - 50), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)    
    cv2.putText(im_out, 'Modified Image', (3 * int(im_out.shape[1] / 4 - 18), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5)
    cv2.putText(im_out, 'Modified Image', (3 * int(im_out.shape[1] / 4 - 18), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(im_out, 'Prediction: ' + activity_classes[pred], (10, im.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5)
    cv2.putText(im_out, 'Prediction: ' + activity_classes[pred], (10, im.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return im_out
