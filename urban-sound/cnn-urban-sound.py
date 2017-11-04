import glob
import matplotlib.pyplot as plt

plt.style.use('ggplot')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield  int(start), int(start + window_size)
        start += (window_size / 2)


def extract_features(parent_dir, sub_dirs, file_ext="*.wav", bands=60, frames=41):
    window_size = 512 * (frames -1)
    log_specgrams = []
    labels = []
    for l, sub_dirs in enumerate(sub_dirs):
        for fn in glob.glob()