from setuptools import find_packages
from distutils.core import setup

setup(name="pytvfemd",
      version="0.1.post1",
      author="Stefano Bianchi",
      author_email="stefanobianchi9@gmail.com",
      url="https://github.com/stfbnc/pytvfemd.git",
      license="GPLv3.0",
      description="tvfemd in Python",
      long_description=open("README.md").read(),
      packages=find_packages(),
      classifiers=["License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python :: 3.5",
                   "Programming Language :: Python :: 3.6",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.8",
                   "Topic :: Scientific/Engineering"],
      python_requires=">=3.5",
      install_requires=["numpy>=1.15"],
      project_urls={"Bug Reports": "https://github.com/stfbnc/pytvfemd/issues",
                    "Source": "https://github.com/stfbnc/pytvfemd/"},
      package_data={"pytvfemd": ["LICENSE"]},
      include_package_data=True
)
