import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

VERSION = '0.1.8'
DESCRIPTION = 'A simplistic and efficient pure-python neural network library that allows to build multilayer neural network with ease.'

setup(
    name = "ak_pynn",
    version=VERSION,
    author="Ankit kohli",
    author_email="<contact.ankitkohli@gmail.com>",
    description=DESCRIPTION,
    license = "MIT",
    packages=find_packages(exclude=['datasets']),
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    keywords = ["neural network", "pure python", "ankit_nn", "machine learning", "ML", "deep learning", "deepL", "MLP", "perceptron","ankit kohli","ak_pynn"],
    install_requires=['numpy', 
                      'tqdm',
                      'opt_einsum',
                      'matplotlib',
                      'numexpr',
                      'seaborn',
                      ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        'Programming Language :: Python :: 3.0',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
    ],
)