import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fenics_helpers",
    version="0.1",
    author="Christoph Pohl",
    author_email="christoph.pohl@bam.de",
    description="A tiny package to modify the node order of quadratic triangles and tets in a xdmf-hdf mesh to work with FEniCS/DOLFIN.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nutofem/fenics_helpers",
    packages=["fenics_helpers", "fenics_helpers.timestepping"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
