from setuptools import setup

__version__ = '0.1'

install_requires = []
setup_requires = []
tests_require = []

setup(
    name='parallel-tacotron-2',
    version=__version__,
    description='parallel tacotron 2 (wip)',
    author='ntt123',
    author_email='ntt123@home',
    keywords=['text-to-speech', 'tacotron', 'parallel-tacotron', 'pytorch', 'spectrogram', 'speech-synthesis'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=['pt2'],
    python_requires='>=3.6',
)
