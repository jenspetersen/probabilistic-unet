from setuptools import setup

setup(name='probunet',
      version='0.1',
      description='',
      url='https://github.com/jenspetersen/probabilistic-unet',
      author='Jens Petersen',
      author_email='jens.petersen@dkfz.de',
      license='MIT',
      packages=['probunet'],
      install_requires=['matplotlib', 'numpy', 'torch', 'trixi', 'batchgenerators'],
      zip_safe=False)
