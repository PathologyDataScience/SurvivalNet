try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os
from pkg_resources import parse_requirements

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('LICENSE') as f:
    license_str = f.read()

try:
    with open('requirements.txt') as f:
        ireqs = parse_requirements(f.read())
except SyntaxError:
    raise
requirements = [str(req) for req in ireqs]

setup(name='survivalnet',
      version='0.1.0',
      description='Deep learning survival models',
      author='Emory University',
      author_email='lee.cooper@emory.edu',
      url='https://github.com/cooperlab/SurvivalNet',
      packages=['survivalnet'],
      package_dir={'survivalnet': 'survivalnet'},
      include_package_data=True,
      install_requires=requirements,
      license=license_str,
      zip_safe=False,
      keywords='survivalnet',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: Console',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
)
