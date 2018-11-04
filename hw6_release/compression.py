import numpy as np

from numpy.linalg import svd
def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size
    
    u,s,vh = svd(image)
    s[num_values:] = 0 #size min(m, n)
    m,n = image.shape
    k = min(m,n)
    #mxm, mxn, nxn
    s_m_n = np.zeros_like(image)
    s_m_n[:k,:k] = np.diag(s)
    compressed_image = np.dot(np.dot(u, s_m_n), vh) #u, vh do not change
    
    #I found an interesting link: https://www.math.cuhk.edu.hk/~lmlui/CaoSVDintro.pdf
    #Let see in page number 4
    compressed_size = num_values * (m + n + 1)
    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size
