import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

def create_montage(image, gt, mask, alpha=0.5, show_plot=True, model_dir=None, image_name=None):
    """Plots image, mask, and overlay
    
    VocalMat segmentation has two classes [0,1], where 0 is background and 1 is vocalization/noise
    
    Arguments:
        image {PIL.Image} -- original image
        mask {PIL.Image} -- ground truth mask
    
    Keyword Arguments:
        alpha {number} -- alpha value to blend image and mask (default: {0.5})
    """
    img      = np.array(image) # torch to numpy
    # img    = np.transpose(img,(1,2,0)) # (N, H, W)->(H, W, N) reshape to plot image
    img      = img[0]

    msk      = np.array(mask)
    msk      = msk[0]
    msk      = (msk > 0.5) * 255.0
    blend    = img * msk

    gt       = np.array(gt)
    gt       = gt[0]
    gt       = (gt > 0) * 255.0
    blend_gt = img * gt

    plt.figure()
    
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('image')
    
    plt.subplot(232)
    plt.imshow(msk, cmap='jet')
    plt.title('mask')
    
    plt.subplot(233)
    plt.imshow(gt, cmap='jet')
    plt.title('ground truth')

    plt.subplot(234)
    plt.imshow(img, cmap='gray')
    plt.imshow(msk, cmap='jet', alpha=alpha)
    plt.title('overlay mask')

    plt.subplot(235)
    plt.imshow(img, cmap='gray')
    plt.imshow(gt, cmap='jet', alpha=alpha)
    plt.title('overlay gt')
    
    plt.subplot(236)
    plt.imshow(blend, cmap='gray')
    plt.title('image*mask')

    if show_plot is True:
        plt.show()
    if model_dir is not None and image_name is not None:
        img = Image.fromarray(img*255).convert("L")
        img.save(os.path.join(model_dir, image_name + '_img.png'))

        msk = Image.fromarray(msk).convert("L")
        msk.save(os.path.join(model_dir, image_name + '_mask.png'))

        plt.savefig(os.path.join(model_dir, image_name + '.png'))
        plt.close()