import numpy as np
import math as math
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.feature_extraction import image
import skimage as sk
from scipy.special import logsumexp


def exp_normalize(x):
    b = x.max(1)
    dif = x - b[:, np.newaxis]
    y = np.exp(dif)
    s = y.sum(1)
    return y / s[:, np.newaxis]


def estimate_noise(I):

  H, W = I.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return sigma

def softmax(A):
    m = A.max(1)
    B = np.exp(A.T - m)
    return (B / (np.sum(B, 0) + np.finfo(np.double).tiny)).T


def comp_psnr(im1, im2):
    res = -10 * math.log(np.var(np.reshape(im1, (-1, 1)) - np.reshape(im2, (-1, 1))), 10)
    return res


def soft_maximum(A):
    m = A.max(1)
    return np.log(np.sum(np.exp(A.T - m), 0) + np.finfo(np.double).tiny) + m


class g_distr:

    def __init__(self, X):
        #self.mu = np.random.rand(1, X.shape[1]).astype(np.float32)
        self.mu = X[np.random.randint(X.shape[0], size=1), :]
        temp = 0.5 + 0.3*np.random.rand(X.shape[1],1)
        self.cov = np.diagflat(temp * 0.95 * np.max(np.diag(np.dot(X.T, X)/X.shape[0])))
        self.pi = 1 / X.shape[1]

        self.V, self.L, c = np.linalg.svd(self.cov)
        self.L = np.maximum(self.L, 1e-10)
        self.L, self.V = self.L[::-1], self.V[:, ::-1]

    def update(self, X, w, sigma):

        self.mu = np.dot(w, X) / np.sum(w)
        #self.mu = np.zeros(self.mu.shape)
        self.cov = np.dot((X - self.mu).T * w, X - self.mu) / np.sum(w)
        self.pi = np.sum(w) / X.shape[0]
        #self.L, self.V = np.linalg.eigh(self.cov)

        self.V, self.L, c = np.linalg.svd(self.cov)

        self.L = np.maximum(self.L - sigma**2, 1e-10)
        self.L, self.V = self.L[::-1], self.V[:, ::-1]
        self.cov = np.dot(self.V, np.dot(np.diag(self.L), self.V.T))

    def log_likelihood(self, X, sigma = 0):
        L, V = self.L + sigma**2, self.V
        A = np.dot(V, np.diag(np.maximum(L, 1e-10) ** -0.5))

        logdet = np.sum(np.log(np.maximum(L, 1e-10)))
        maha = np.sum(np.dot(X - self.mu, A) ** 2, 1)

        p = self.cov.shape[0]
        a = - 0.5 * p * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * maha

        return a

    def grad_log_likelihood(self, X):
        L, V = self.L, self.V
        A = np.dot(V, np.diag(np.maximum(L, 1e-10) ** -0.5))

        logdet = np.sum(np.log(np.maximum(L, 1e-10)))
        maha = np.sum(np.dot(X - self.mu, A) ** 2, 1)

        p = self.cov.shape[0]
        b = - 0.5 * p * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * maha

        return b




class gmm:

    def __init__(self, verbose=True):
        self.verbose = verbose

    def get_pi(self):
        pi = np.asarray([ci.pi for ci in self.components])
        return pi / np.sum(pi)

    def log_likelihood(self, X, sigma = 0):
        Z = np.zeros((X.shape[0], len(self.components)))
        for j, cj in enumerate(self.components):
            Z[:, j] = cj.log_likelihood(X, sigma)
        return np.mean(soft_maximum(Z + np.log(self.get_pi())))

    def fit(self, X, N_comp=20, N_iter=100, sigma=0):

        self.N = X.shape[0]
        self.p = X.shape[1]

        self.components = [g_distr(X) for i in range(N_comp)]
        indic = np.random.rand(self.N, N_comp).astype(np.float32)
        #Z = (Z.T / np.sum(Z, 1)).T

        for i in range(N_iter):
            for j, cj in enumerate(self.components):
                indic[:, j] = np.log(cj.pi) + cj.log_likelihood(X, sigma)
            #Z = (indic.T / (np.sum(indic, 1) + np.finfo(np.double).tiny)).T
            Z = exp_normalize(indic)
            #Z = softmax(indic)
            for j, cj in enumerate(self.components):
                cj.update(X, Z[:, j], sigma)
            if self.verbose == True:
                print('iteration ' + str(i + 1) + ': ' + str(np.mean(soft_maximum(indic))))

            #Z = softmax(Z)


verbose = False

img_orig=mpimg.imread('Cameraman256.png')

pd = 6

img = sk.util.pad(img_orig, (pd, pd), 'symmetric')

imSize = img.shape

sigma = 25/255

np.random.seed(0)

noise = np.random.normal(0, 1, imSize)

noisy = img + sigma*noise

sigmaHat = estimate_noise(noisy)

psnr_in = comp_psnr(img[pd:-pd, pd:-pd], noisy[pd:-pd, pd:-pd])

print(psnr_in)

patches = image.extract_patches_2d(noisy, (pd,pd))


a = patches.shape
b = (a[0],a[1]*a[2])
newshape = tuple(b)
patches = np.reshape(patches, newshape)

patches_dc = patches.mean(1)
patches_ac = patches - patches_dc[:, np.newaxis]

N_iter = 100
N_comp = 20
# fit model:
model1 = gmm(verbose)
model1.fit(patches_ac, N_comp, N_iter, sigma)

temp = np.zeros(patches_ac.shape)

py = np.zeros((patches_ac.shape[0], N_comp))
for j, cj in enumerate(model1.components):
    py[:,j] = np.log(cj.pi) + cj.log_likelihood(patches_ac, sigma)

py = exp_normalize(py)

for j, cj in enumerate(model1.components):
    filter = np.dot(cj.cov, np.linalg.inv(cj.cov + sigma**2 * np.diag(np.ones(pd**2))))
    aux = np.dot(filter, patches_ac.transpose(1,0)) + np.dot(filter, np.dot(np.dot(sigma**2 * np.diag(np.ones(pd**2)), np.linalg.inv(cj.cov)), cj.mu[:, np.newaxis]))
    #aux = np.dot(filter + np.diag(np.ones(pd**2)), (cj.mu[:, np.newaxis] + np.dot(np.dot(cj.cov, 1 / sigma**2 * np.diag(np.ones(pd**2))), patches_ac.transpose(1,0))))
    temp = temp + aux.transpose(1, 0) * py[:, np.newaxis, j]

x_hat_patches = temp

# evaluate:
#print(model1.log_likelihood(patches_ac, sigma))


x_hat_patches = np.reshape(np.clip(x_hat_patches + patches_dc[:, np.newaxis], 0, 1), a)

x_hat1 = image.reconstruct_from_patches_2d(x_hat_patches, imSize[0:2])

#plt.imshow(x_hat1, cmap = 'gray')
#


psnr_out1 = comp_psnr(img_orig, x_hat1[pd:-pd, pd:-pd])

print(psnr_out1)




