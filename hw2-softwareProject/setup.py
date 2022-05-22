from setuptools import setup,find_packages,Extension

setup(
    name='mykmeanssp',
    version = '0.1.0',
    author = 'Nir',
    description="t",
    install_requires=['invoke'],
    packages=find_packages(),
    license='GPL-2',
    ext_modules=[
        Extension(
            'mykmeanssp',
            ['kmeans.c'],
        ),
    ]
)