from setuptools import setup, find_packages

setup(
    name='quantum_pulse_simulator',        # Unique name for your package
    version='0.1.0',                       # Version number
    packages=find_packages(where='src'),   # Automatically find packages in src/
    package_dir={'': 'src'},               # Root package directory is src/
)
