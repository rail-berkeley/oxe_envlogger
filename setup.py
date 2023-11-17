from setuptools import setup, find_packages

setup(
    name='oxe_envlogger',
    version='0.0.1',
    description='library to log env to tfds',
    url='https://github.com/rail-berkeley/oxe_envlogger',
    author='auth',
    author_email='tan_you_liang@hotmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'gym',
        'typing',
        'dm-env',
        'envlogger[tfds]',
        "tensorflow>=2.10.0",
        "tensorflow_datasets>=4.8.0",
    ],
    zip_safe=False
)
