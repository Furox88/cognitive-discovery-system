from cds.signals import dft, fft_radix2, convolve, low_pass_filter

# FFT of a signal
signal = [complex(i) for i in range(8)]
spectrum = fft_radix2(signal)

# Convolution
result = convolve([1.0, 2.0, 3.0], [0.5, 0.5])
print(result)  # [0.5, 1.5, 2.5, 1.5]
