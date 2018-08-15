from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup()
d['packages'] = ['dougsm_helpers']
d['package_dir'] = {'': 'src'}

setup(**d)
