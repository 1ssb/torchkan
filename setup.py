from setuptools import setup, find_packages

setup(
    name='TorchKAN',  # Replace with your package name
    version='0.1.0',  # Version 0 indicating early development
    packages=find_packages(),
    description='An easy-to-use PyTorch implementation of the Kolmogorov Arnold Network',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Subhransu Bhattacharjee',
    author_email='Subhransu.Bhattacharjee@anu.edu.au',
    url='https://github.com/1ssb/torchkan',
    install_requires=[
        'torch==1.8.1+cu101',
        'torchvision==0.9.1+cu101',
        'torchaudio==0.8.1',
        'wandb',
        'tqdm',
        'numpy',
        'matplotlib'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Libraries',
    ]
)
