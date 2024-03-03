from setuptools import find_packages
from distutils.core import setup

setup(name="pytvfemd",
      version="0.3.14",
      author="Stefano Bianchi, Alessandro Longo",
      author_email="stefanobianchi9@gmail.com",
      url="https://github.com/stfbnc/pytvfemd.git",
      license="GPLv3.0",
      description="tvfemd algorithm in Python",
      long_description=open("README.md").read(),
      packages=find_packages(),
      classifiers=["License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python :: 3",
                   "Topic :: Scientific/Engineering"],
      python_requires=">=3.8",
      install_requires=["numpy",
                        "scipy"],
      project_urls={"Bug Reports": "https://github.com/stfbnc/pytvfemd/issues",
                    "Source": "https://github.com/stfbnc/pytvfemd/"},
      package_data={"pytvfemd": ["LICENSE"]},
      include_package_data=True
      )
