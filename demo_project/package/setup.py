from setuptools import setup

setup(
    name='package',
    version='0.1',
    description = 'MyPackage',
    author="David Lee",
    author_email="youngenie9@gmail.com",
    packages=['package.feature', 'package.ml_training', "package.utils"],
    install_requires=['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'mlflow']
)