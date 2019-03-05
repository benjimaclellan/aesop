from setuptools import setup

"""
Setup information for package distribution
"""

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
                'uuid'
                ],
      zip_safe=False)
