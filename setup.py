from setuptools import setup, find_packages

setup(
   name='cempy',
   version='0.0.1',
   author='Enayetur Raheem',
   author_email='eraheem@dataskool.com',
   packages=find_packages(),
   url='http://github.com/raheems/cempy',
   license='LICENSE.md',
   description='Coarsened Exact Matching with Python',
   long_description=open('cempy/README.md').read()
)