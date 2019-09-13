try:
    from setuptools import setup

except ImportError:
    from disutils.core import setup

config = {
    'description':'proj5',
    'author':'Paul de Fusco',
    'url':'URL to get it at.',
    'download_url':'Where to download it.',
    'author_email':'pauldefusco@gmail.com',
    'version':'0.1',
    'install_requires':['nose'],
    'packages':['proj1'],
    'scripts':[],
    'name':'proj5'
}

setup(**config)
