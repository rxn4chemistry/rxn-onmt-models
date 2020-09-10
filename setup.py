import io
import re

from setuptools import setup

match = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open('rxn_onmt_utils/__init__.py', encoding='utf_8_sig').read()
)
if match is None:
    raise SystemExit('Version number not found.')
__version__ = match.group(1)

setup(
    name='rxn_onmt_utils',
    version=__version__,
    author='IBM RXN team',
    packages=['rxn_onmt_utils'],
    package_data={'rxn_onmt_utils': ['py.typed']},
    install_requires=['attrs>=19.1.0', 'OpenNMT-py>=1.0.0']
)
