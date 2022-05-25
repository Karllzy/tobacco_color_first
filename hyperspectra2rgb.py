import numpy as np
import pandas as pd
from scipy.interpolate import interpolate


def standard_cie1964d10(raw_spectra: np.ndarray, data_wavelength: np.ndarray,
                     cie_file: str = 'config/CIE1964d10.csv') -> np.ndarray:
    """
    Convert spectra to RGB image.

    :param raw_spectra: raw spectra img (h, w, channels)
    :param data_wavelength: wavelength of raw spectra (channels, )
    :param cie_file: config of CIE1964
    :return: RGB img
    """
    cie1964d10 = pd.read_csv(cie_file, header=0, index_col=0)
    cie1964d10 = cie1964d10.loc[400: 680]
    weights = cie1964d10.values
    data_collected_for_cie = []
    for wavelength in cie1964d10.index:
        error_vect = np.abs(data_wavelength - wavelength)
        required_idx = np.argmin(error_vect)
        error = np.min(error_vect)
        if error > 1:
            print(f"{wavelength}nm get suitable spectra channel failed,"
                  "the error is {error}")
        data_collected_for_cie.append(raw_spectra[:, :, required_idx][..., np.newaxis])
    data_collected_for_cie = np.concatenate(data_collected_for_cie, axis=2)
    return spec_weight_rgb(data_collected_for_cie, weights)


def spec_weight_rgb(spectra, weight, suppress_r=False):
    xyz_img = np.dot(spectra, weight)
    matrix_m = np.array([[2.7688, 1.7517, 1.1301], [1, 4.5906, 0.0601], [0, 0.0565, 5.5942]])
    rgb_img = np.dot(xyz_img, np.linalg.inv(matrix_m))
    if suppress_r:
        rgb_img[:, :, 0] *= 0.293
    rgb_img = rgb_img / rgb_img.max()
    rgb_img = np.array(rgb_img * 255, dtype=np.uint8)
    return rgb_img


def force_hyper_rgb(raw_spectra: np.ndarray, data_wave_length: np.ndarray,
                    cie_file: str = 'config/CIE1964d10.csv') -> np.ndarray:
    """
    Convert spectra to RGB image.

    :param raw_spectra: raw spectra img (h, w, channels)
    :param data_wave_length: wavelength of raw spectra (channels, 1)
    :param cie_file: config of CIE1964
    :return: RGB img
    """
    cie1964d10 = pd.read_csv(cie_file, header=0, index_col=0)
    cie1964d10 = cie1964d10.loc[400: 680]
    weights = cie1964d10.values
    standard_wave_length = cie1964d10.loc[400:680].index
    dst_wavelength_start, dst_wavelength_end = data_wave_length[0], data_wave_length[-1]
    normalized_point = (standard_wave_length - standard_wave_length[0]) /\
                       (standard_wave_length[-1] - standard_wave_length[0])
    new_data_wavelength = normalized_point * (dst_wavelength_end - dst_wavelength_start) + dst_wavelength_start
    split_weights = [weights[:, i] for i in range(3)]
    split_funcs = [interpolate.interp1d(new_data_wavelength, v, kind='cubic') for v in split_weights]
    new_weights = [func(data_wave_length) for func in split_funcs]
    new_weights = np.concatenate(new_weights, axis=1)
    return spec_weight_rgb(raw_spectra, new_weights)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    raw_file_name, raw_file_shape = "/home/share/dataset/calibrated34.raw", (600, 1024, 448)
    data_hdr = "../share/dataset/wavelength.csv"
    with open(raw_file_name, "rb") as f:
        raw_spectra = np.frombuffer(f.read(), dtype=np.float32).reshape((600, 448, 1024)).transpose(0, 2, 1)
    data_wavelength = pd.read_csv(data_hdr, header=None, index_col=None).values
    rgb = force_hyper_rgb(raw_spectra, data_wavelength)
    cv2.imwrite("/home/zhenye/tobacco_color_first/assets/", rgb[..., ::-1])


