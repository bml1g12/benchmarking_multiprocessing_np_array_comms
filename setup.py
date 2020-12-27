"""Optionally install benchmarks as a module"""
#!/usr/bin/env python

import setuptools

setuptools.setup(name="array_benchmark",
                 version="X",
                 description="Benchmarking sharing numpy arrays",
                 author="Scoville",
                 python_requires=">=3.7",
                 packages=setuptools.find_packages(),
                )
