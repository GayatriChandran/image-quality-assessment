#!/usr/bin/env python
"""
Finds the 2D fourier transform of an image.

Gayatri 01/20
"""

import matplotlib.pyplot as plt
from scipy import fftpack
import numpy
import tifffile


if (__name__ == "__main__"):

    image = tifffile.imread('images/emgain_0006.tif')
    M, N = image.shape

    F = numpy.abs(fftpack.fftshift(fftpack.fft2(image)))**2

    print('Std = ', F.var())

    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(image)
    axs[0].set_title('Actual image')
    
    axs[1].imshow(numpy.log(1 + F), cmap='viridis',
          extent=(-N // 2, N // 2, -M // 2, M // 2))
    axs[1].set_title('Power Spectrum')

    # Set block around center of spectrum to zero
    K = 1
    F1 = F.copy()
    F1[M // 2 - K: M // 2 + K, N // 2 - K: N // 2 + K] = 0

    # Find all peaks higher than the 98th percentile
    peaks = F1 < numpy.percentile(F1, 98)

    # Shift the peaks back to align with the original spectrum
    peaks = fftpack.ifftshift(peaks)

    # Make a copy of the original (complex) spectrum
    F_dim = F1.copy()

    # Set those peak coefficients to zero
    F_dim = F_dim * peaks.astype(int)

    # Do the inverse Fourier transform to get back to an image.
    # Since we started with a real image, we only look at the real part of
    # the output.

    image_filtered = numpy.real(fftpack.ifft2(F_dim))

    axs[2].imshow(numpy.log10(1 + numpy.abs(F_dim)), cmap='viridis')
    axs[2].set_title('Suppressed Spectrum')

    axs[3].imshow(image_filtered)
    axs[3].set_title('Reconstructed image')

    # plt.show()
    