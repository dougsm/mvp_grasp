import numpy as np
import cv2


def gridshow(name, imgs, scales, cmaps, columns, border=10, show=False):
    """
    Display images in a grid.
    :param name: cv2 Window Name
    :param imgs: List of np.ndarray images (2 or 3D)
    :param scales: List of tuples (min, max) values to scale the image or None
    :param cmaps: cv2 color map to apply (for 2D imgs)
    :param columns: Number of columns to display
    :param border: Border between images
    :param show: if True, then call cv2.imshow
    :return: 3D np.ndarray to display
    """
    imgrows = []
    imgcols = []

    maxh = 0
    for i, (img, cmap, scale) in enumerate(zip(imgs, cmaps, scales)):
        if scale is not None:
            img = (np.clip(img, scale[0], scale[1]) - scale[0])/(scale[1]-scale[0])
        elif img.dtype == np.float:
            img = (img - img.min())/(img.max() - img.min())
        if cmap is not None:
            imgc = cv2.applyColorMap((img * 255).astype(np.uint8), cmap)
        else:
            imgc = img

        maxh = max(maxh, imgc.shape[0])
        imgcols.append(imgc)

        if i > 0 and i % columns == (columns - 1):
            imgrows.append(np.hstack([np.pad(c, ((0, maxh - c.shape[0]), (border//2, border//2), (0, 0)), mode='constant') for c in imgcols]))
            imgcols = []
            maxh = 0

    # Unfinished row
    if imgcols:
        imgrows.append(np.hstack([np.pad(c, ((0, maxh - c.shape[0]), (border//2, border//2), (0, 0)), mode='constant') for c in imgcols]))

    maxw = max([c.shape[1] for c in imgrows])

    if show:
        cv2.imshow(name, np.vstack([np.pad(r, ((border//2, border//2), (0, maxw - r.shape[1]), (0, 0)), mode='constant') for r in imgrows]))
    return np.vstack([np.pad(r, ((border//2, border//2), (0, maxw - r.shape[1]), (0, 0)), mode='constant') for r in imgrows])
