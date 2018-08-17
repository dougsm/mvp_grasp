from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup()
d['packages'] = ['mvp_grasping']
d['package_dir'] = {'': 'src'}

setup(**d)
