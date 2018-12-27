from setuptools import setup, find_packages
setup(name='persistent_doc',
      version='0.5.2',
      description='An XML-like document with spreadsheet formulas for values and underlying persistent data structures',
      long_description=open("readme.md").read(),
      long_description_content_type="text/markdown",
      url='https://github.com/asrp/persistent_doc',
      author='asrp',
      author_email='asrp@email.com',
      packages=find_packages(),
      install_requires=['pyrsistent'],
      keywords='persistent tree document formula')
