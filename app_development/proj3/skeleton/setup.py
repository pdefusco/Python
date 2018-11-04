try:
    from setuptools import setup

except ImportError:
    from disutils.core import setup

config = {
    'description':'project 3',
    'author':'Paul de Fusco',
    'url':'URL to get it at.',
    'download_url':'Where to download it.',
    'author_email':'My Email',
    'version':'0.1',
    'install_requires':['nose'],
    'packages':['proj3'],
    'scripts':[],
    'name':'proj3'
}

setup(**config)
