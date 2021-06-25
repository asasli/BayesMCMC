from setuptools import setup, version
with open("README.md") as f:
    long_description = f.read()
    
setup(
    name='BayesMCMC',
    version='1.0',
    description='MCMC method + Bayesian Statistics to estimate the accuracy of distance measurments',
    license="MIT",
    long_description=long_description,
    author=('Argyro Sasli','Isaac Mutie'),
    author_email=('asasli@auth.gr','mumoisaac@gmail.com'),
    packages=['BayesMCMC'],
    modules=['resources'],
    classifiers=['Development Status :: 4 - Beta',
    'License :: OSI Approved :: MIT license'
    ]

)
