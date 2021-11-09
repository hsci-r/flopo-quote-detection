from setuptools import setup, find_packages

with open('README.md') as fp:
    README = fp.read()

setup(
    name='flopo-quote-detection',
    version='0.1.0',
    author='Maciej Janicki',
    author_email='maciej.janicki@helsinki.fi',
    description='Rule-based quote detection for Finnish news.',
    long_description=README,
    long_description_content_type='text/markdown',
    packages=find_packages('src', exclude=['tests', 'tests.*']),
    package_dir={'': 'src'},
    package_data={'flopo_quote_detection': ['rules_*.yaml']},
    test_suite='tests',
    install_requires=['pyyaml', 'spacy'],
    entry_points={
        'console_scripts' : [
            'find_actors   = flopo_quote_detection.find_actors:main',
            'find_quotes   = flopo_quote_detection.find_quotes:main',
        ]
    }
)
