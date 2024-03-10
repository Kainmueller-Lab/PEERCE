from setuptools import setup, find_packages

setup(
    name='apedia',
    version='0.1.0',
    author='Your Name',
    author_email='Fabian.Reith@charite.de',
    packages=find_packages(),
    description='A deep learning tool for enhancing diagnostics of PD-L1 expression in angiosarcoma',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy', 
        'pandas', 
        'torch', 
        # TODO Add more dependencies
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        # TODO Add more classifiers (https://pypi.org/classifiers/)
    ],
)
