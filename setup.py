import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="e2e-Dutch",
    version="0.0.1",
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
    python_requires='>=3.6',
)