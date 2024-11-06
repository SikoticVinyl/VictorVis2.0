from setuptools import setup, find_packages

setup(
    name="grid_collector",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'gql',
        'python-dotenv',
        'pandas',
        'requests_toolbelt'
    ]
)
