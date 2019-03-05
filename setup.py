from setuptools import setup

setup(name='asope',
      version='0.1',
      description='Automated seach for optical experiments',
      url='https://github.com/benjimaclellan/ASOPE',
      author='Benjamin MacLellan',
      author_email='benjamin.maclellan@emt.inrs.ca',
      license='Copyright Benjamin MacLellan',
      install_requires=[
                'numpy',
                'networkx', 
                'deap', 
                'peakutils', 
                'matplotlib', 
                'multiprocess',
                'uuid',
                'beeprint',
                ],
      zip_safe=False)

"""
Don't need to have the following packages in the install_requires (standards):
    time
    pickle
    
"""

#from distutils.core import setup
#
#setup(
#    # Application name:
#    name="asope",
#
#    # Version number (initial):
#    version="0.1.0",
#
#    # Application author details:
#    author="Benjamin MacLellan",
#    author_email="benjamin.maclellan@emt.inrs.ca",
#
#    # Packages
#    packages=["asope"],
#
#    # Include additional files into the package
#    include_package_data=True,
#
#    # Details
#    url="https://github.com/benjimaclellan/ASOPE",
#
#    #
#    # license="LICENSE.txt",
#    description="Automated seach for optical experiments.",
#
#    # long_description=open("README.txt").read(),
#
#    # Dependent packages (distributions)
#    install_requires=[  'networkx',
#                        'deap',
#                        'numpy', 
#                        'peakutils', 
#                        'matplotlib', 
#                        'multiprocess',
#                        'time',
#                        'uuid',
#                        'pickle',
#                        'beeprint',
#                        ],
#)