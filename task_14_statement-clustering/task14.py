from skimage.io import imread, imsave
from skimage import img_as_float
import pandas as pd
import numpy
import math
from sklearn.cluster import KMeans

image = img_as_float(imread('parrots.jpg'))
n, m, rgb = image.shape
X = pd.DataFrame(numpy.reshape(image, (n*m, rgb)), columns=['R', 'G', 'B'])

def clustering(matrix, n_clusters):
    print('Clustering: ' + str(n_clusters))
    matrix = matrix.copy()
    clf = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)
    matrix['cluster'] = clf.fit_predict(matrix)

    means = matrix.groupby('cluster').mean().values
    #means_pixels = [means[i] for i in matrix['cluster'].values]
    means_image = numpy.reshape([means[i] for i in matrix['cluster'].values], (n, m, rgb))
    imsave('means/means_image_' + str(n_clusters) + '.jpg', means_image)

    medians = matrix.groupby('cluster').median().values
    #medians_pixels = [medians[i] for i in matrix['cluster'].values]
    medians_image = numpy.reshape([medians[i] for i in matrix['cluster'].values], (n, m, rgb))
    imsave('medians/medians_image_' + str(n_clusters) + '.jpg', medians_image)

    return means_image, medians_image

def psnr(image1, image2):
    mse = numpy.mean((image1 - image2) ** 2)
    return 10 * math.log10(float(1) / mse)

for j in range(1, 21):
    means_image, medians_image = clustering(X, j)
    psnr_mean, psnr_median = psnr(image, means_image), psnr(image, medians_image)
    print(psnr_mean, psnr_median)

    if psnr_mean > 20 or psnr_median > 20:
        print(j)
        break
