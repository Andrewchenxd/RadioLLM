import torch
import numpy as np
import einops


def filter(sgn,filiter='high',filiter_threshold=0.99,filiter_size=0.0,middle_zero=False,freq_smooth=False,return_IQ=False):
    I = sgn[0]
    Q = sgn[1]
    IQ = I + 1j * Q  # 复数形式的IQ数据

    # 对复数数据进行傅里叶变换
    N = len(IQ)
    IQ_fft = np.fft.fft(IQ,n=N)
    IQ_abs = np.abs(IQ_fft)
    sorted_indices = np.argsort(IQ_abs)
    if filiter=='high':
        threshold_index = int(filiter_threshold * N)  # 计算排名前20%的索引
        threshold = IQ_abs[sorted_indices[threshold_index]]  # 找到阈值
        IQ_fft[IQ_abs >= threshold] *= filiter_size  # 将大于阈值的点设为0
    elif filiter == 'low':
        threshold_index = int(filiter_threshold * N)  # 计算排名前20%的索引
        threshold = IQ_abs[sorted_indices[threshold_index]]  # 找到阈值
        IQ_fft[IQ_abs <= threshold] *= filiter_size  # 将大于阈值的点设为0
        if middle_zero:
            IQ_fft[20:110]=0.001
        if freq_smooth:
            # 定义平滑窗口的大小
            window_size = 3

            # 创建一个新的数组，用于存储平滑后的数据
            smoothed_arr = np.zeros_like(IQ_fft)

            # 对数组进行平滑处理
            for i in range(window_size, len(IQ_fft) - window_size):
                smoothed_arr[i] = np.mean(IQ_fft[i - window_size:i + window_size])
    sgn_IQ = np.copy(sgn)
    sgn = np.fft.ifft(IQ_fft)
    if return_IQ:
        sgn_IQ[0]=np.real(sgn)
        sgn_IQ[1] = np.imag(sgn)
        return sgn_IQ
    else:
        return sgn

def sgn_norm(sgn,normtype='maxmin'):
    if normtype=='maxmin':
        sgn = (sgn - sgn.min()) / (sgn.max() - sgn.min())
    elif normtype == 'maxmin-1':
        sgn = (2*sgn - sgn.min()- sgn.max()) / (sgn.max() - sgn.min())
    else:
        sgn=sgn
    return sgn


def moving_avg_filter(signal, window_size):
    """
    Applies a moving average filter to the input signal.

    Args:
        signal (numpy.ndarray): Input signal with shape (2, L), where 2 represents I/Q channels and L is the signal length.
        window_size (int): Size of the moving average window.

    Returns:
        numpy.ndarray: Filtered signal with the same shape as the input signal.
    """
    # Check if the input signal has the expected shape
    if signal.shape[0] != 2:
        raise ValueError("Input signal must have shape (2, L), where 2 represents I/Q channels.")

    filtered_signal = np.zeros_like(signal)

    # Apply the moving average filter to each channel
    for channel in range(2):
        # Pad the signal with zeros to handle boundary conditions
        padded_signal = np.pad(signal[channel], (window_size // 2, window_size - 1 - window_size // 2), mode='edge')

        for i in range(signal.shape[1]):
            # Calculate the moving average using the window
            filtered_signal[channel, i] = np.sum(padded_signal[i:i + window_size]) / window_size

    return filtered_signal

def sgn_freq(sgn,freq_choose='fft',window='None'):
    # freq = sgn[0, :] + 1j * sgn[1, :]
    # freq = torch.tensor(freq, dtype=torch.float32)
    I = sgn[0, :]
    Q = sgn[1, :]
    I = torch.tensor(I, dtype=torch.float32)
    Q = torch.tensor(Q, dtype=torch.float32)
    if freq_choose =='fft':
        sgn = torch.abs(torch.fft.fft(I+Q*1j))
    elif freq_choose =='stft':
        if window=='None':
            Window=None
            sgn = torch.abs(torch.stft(I + Q * 1j, n_fft=128, hop_length=5, win_length=8, window=Window))
        elif window=='hanming':
            Window=torch.hann_window(8)
            sgn = torch.abs(torch.stft(I + Q * 1j, n_fft=128, hop_length=5, win_length=8,window=Window))
        elif window=='None_IQ':
            Window=None
            sgn = torch.abs(torch.stft(I + Q, n_fft=128, hop_length=5, win_length=8, window=Window))

    return sgn

def gaussian_filter(signal, sigma=1, kernel_radius=7):
    """
    Applies a Gaussian filter to the input signal.

    Args:
        signal (numpy.ndarray): Input signal with shape (2, L), where 2 represents I/Q channels and L is the signal length.
        sigma (float): Standard deviation of the Gaussian kernel.
        kernel_radius (int): Radius of the Gaussian kernel.

    Returns:
        numpy.ndarray: Filtered signal with the same shape as the input signal.
    """
    # Check if the input signal has the expected shape
    if signal.shape[0] != 2:
        raise ValueError("Input signal must have shape (2, L), where 2 represents I/Q channels.")

    filtered_signal = np.zeros_like(signal)

    # Create the Gaussian kernel
    x = np.arange(-kernel_radius, kernel_radius + 1)
    gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)

    # Normalize the Gaussian kernel
    # gaussian_kernel /= np.sum(gaussian_kernel)

    # Apply the Gaussian filter to each channel
    for channel in range(2):
        # Pad the signal with zeros to handle boundary conditions
        padded_signal = np.pad(signal[channel], kernel_radius, mode='edge')

        for i in range(signal.shape[1]):
            # Convolve the signal with the Gaussian kernel
            filtered_signal[channel, i] = np.convolve(padded_signal, gaussian_kernel, mode='valid')[i]

    return filtered_signal