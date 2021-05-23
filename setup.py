from distutils.core import setup


with open('README.md') as file:
    long_description = file.read()

setup(
    name='path-homology-toolkit',
    packages=['path_homology'],
    version='0.1.3',
    description='The algorithm for computing path homology',
    author='Maksim Nekrashevich',
    author_email='mvnekrashevich@edu.hse.ru',
    url='https://github.com/max-nekrashevich/path-homology-toolkit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 1 - Production/Stable",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Environment :: Other Environment",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    python_requires=">=3.9"

)