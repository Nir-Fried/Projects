from setuptools import setup,find_packages,Extension

setup(
    name='spkmeans',
    version = '0.1.0',
    author = 'ABC',
    description="t",
    install_requires=['invoke'],
    packages=find_packages(),
    license='GPL-2',
    ext_modules=[
        Extension(
            'spkmeans',
            ['spkmeansmodule.c'],
        ),
    ]
)