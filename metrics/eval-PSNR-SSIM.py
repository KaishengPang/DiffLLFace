import cv2
import numpy as np
import math
import os
import torch


def calc_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calc_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('SAME SHAPE ERROR')
    
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('ERROR')

def rgb2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img = img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0],
                              [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
 
def calc_metrics(img1, img2, crop_border=8, test_Y=True):
    if img1.shape != img2.shape:
        raise ValueError('ERROR shapes: {} and {}'.format(img1.shape, img2.shape))
    
    if img1.ndim == 3 and img1.shape[2] == 3:  
        if test_Y:
            img1_y = rgb2ycbcr(img1, only_y=True)
            img2_y = rgb2ycbcr(img2, only_y=True)
        else:
            img1_y = img1
            img2_y = img2
    elif img1.ndim == 2:  
        img1_y = img1
        img2_y = img2
    else:
        raise ValueError('ERROR')

    
    if crop_border != 0:
        if img1_y.ndim == 3:
            cropped_img1 = img1_y[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_img2 = img2_y[crop_border:-crop_border, crop_border:-crop_border, :]
        elif img1_y.ndim == 2:
            cropped_img1 = img1_y[crop_border:-crop_border, crop_border:-crop_border]
            cropped_img2 = img2_y[crop_border:-crop_border, crop_border:-crop_border]
    else:
        cropped_img1 = img1_y
        cropped_img2 = img2_y
    
    psnr = calc_psnr(cropped_img1 * 255, cropped_img2 * 255)
    ssim = calc_ssim(cropped_img1 * 255, cropped_img2 * 255)
    
    
    return psnr, ssim

def compare_folders(folder_ref, folder_test, crop_border=8, test_Y=True):
    psnr_total = 0
    ssim_total = 0
    count = 0
    
    files_ref = sorted([f for f in os.listdir(folder_ref) if f.endswith('.png')])
    files_test = sorted([f for f in os.listdir(folder_test) if f.endswith('.png')])
    
    if len(files_ref) != len(files_test):
        raise ValueError("number of files in reference and test folders do not match")
    

    for filename in files_ref:
        if filename not in files_test:
            raise ValueError(f"not found {filename} in test folder")
        
        img_ref_path = os.path.join(folder_ref, filename)
        img_test_path = os.path.join(folder_test, filename)
        
        img_ref = cv2.imread(img_ref_path).astype(np.float32) / 255.0
        img_test = cv2.imread(img_test_path).astype(np.float32) / 255.0
        
        if img_ref is None or img_test is None:
            raise ValueError(f"can not load {filename}")

        if img_ref.shape != img_test.shape:
            raise ValueError(f" {filename} shape mismatch")
        
        psnr, ssim = calc_metrics(img_ref, img_test, crop_border=crop_border, test_Y=test_Y)
        psnr_total += psnr
        ssim_total += ssim
        count += 1
        
        print(f"{filename} PSNR：{psnr:.4f}，SSIM：{ssim:.4f}")
    
    average_psnr = psnr_total / count if count != 0 else 0
    average_ssim = ssim_total / count if count != 0 else 0
    print(f"\n average PSNR：{average_psnr:.4f}")
    print(f"average SSIM：{average_ssim:.4f}")


if __name__ == "__main__":
    folder_ref = "/data/pks/DiffLLFace/dataset/CelebA/test/HR"     
    folder_test = "/data/pks/DiffLLFace/output_DiffLLFace"  
    compare_folders(folder_ref, folder_test, crop_border=8, test_Y=True)