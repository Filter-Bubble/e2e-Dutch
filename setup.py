import setuptools
import setuptools.command.build_py
import os
import subprocess
import pkg_resources

__version__ = None

with open("README.md", "r") as fh:
    long_description = fh.read()

with open(os.path.join(os.path.dirname(__file__), 'e2edutch/__version__.py')) as versionpy:
    exec(versionpy.read())


class BuildPyCommand(setuptools.command.build_py.build_py):
    """Build the tensorflow kernels, and proceed with default build."""

    def run(self):
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("This module requires tensorflow to be installed. " +
                              "Install it first with `pip install tensorflow`")
        args = ["g++", "-std=c++11", "-shared"]
        args += [
            pkg_resources.resource_filename("e2edutch", "coref_kernels.cc"),
            "-o",
            pkg_resources.resource_filename("e2edutch", "coref_kernels.so")
        ]
        args += ["-fPIC"]
        args += tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags()
        args += ["-O2",
                 "-D_GLIBCXX_USE_CXX11_ABI=0"]
        subprocess.check_call(args)
        setuptools.command.build_py.build_py.run(self)


setuptools.setup(
    name="e2e-Dutch",
    version=__version__,
    author="Dafne van Kuppevelt",
    author_email="d.vankuppevelt@esciencecenter.nl",
    description="Coreference resolution with e2e for Dutch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Filter-Bubble/e2e-Dutch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={'e2edutch': ["cfg/*.conf", "coref_kernels.so"]},
    cmdclass={
        "build_py": BuildPyCommand
    },
    test_suite='test',
    python_requires='>=3.6',
    install_requires=[
        "tensorflow>=2.0.0",
        "h5py",
        "pyhocon",
        "scipy",
        "scikit-learn",
        "torch<=1.7.1",
        "transformers<=3.5.1",
        "KafNafParserPy",
        "stanza"
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle',
    ],
    extras_require={"doc": ["sphinx",
                            "m2r2"]}
)
