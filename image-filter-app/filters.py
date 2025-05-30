import cv2
import numpy as np

def apply_filter(image, filter_type):
    if filter_type == 'grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    elif filter_type == 'sepia':
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia = cv2.transform(image, sepia_filter)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    
    elif filter_type == 'negative':
        return cv2.bitwise_not(image)
    
    elif filter_type == 'sketch':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return sketch
    
    elif filter_type == 'blur':
        return cv2.GaussianBlur(image, (15, 15), 0)

    elif filter_type == 'cartoon':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 9, 10)
        color = cv2.bilateralFilter(image, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

    elif filter_type == 'emboss':
        kernel = np.array([[ -2, -1, 0],
                           [ -1,  1, 1],
                           [  0,  1, 2]])
        return cv2.filter2D(image, -1, kernel)

    elif filter_type == 'sharpness':
        kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])
        return cv2.filter2D(image, -1, kernel)

    elif filter_type == 'brightness':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 2] = cv2.add(hsv[..., 2], 50)  # Increase brightness
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif filter_type == 'contrast':
        alpha = 1.5 # Contrast control
        beta = 0    # Brightness control
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    elif filter_type == 'noise':
        row, col, _ = image.shape
        gauss = np.random.normal(0, 25, (row, col, 3))
        noisy = np.array(image, dtype=np.float32) + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    elif filter_type == 'vignette':
        rows, cols = image.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, 200)
        kernel_y = cv2.getGaussianKernel(rows, 200)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        vignette = np.copy(image)
        for i in range(3):
            vignette[:, :, i] = vignette[:, :, i] * mask
        return vignette

    else:
        return image
