from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='In this project, a neural decoder is developed which takes time series of brain signals and predicts time series of electromyographic signals of different muscles. This is achieved by building a LSTM decoder for offline decoding, and a basic RNN for mimicking an online decoder.',
    author='Christopher Hitz',
    license='',
)
