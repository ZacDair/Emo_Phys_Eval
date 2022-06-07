from setuptools import setup

# Third Party Dependency Requirements
REQUIRES = ['PyYaml',
            'scikit-learn',
            'pandas',
            'numpy',
            'shap',
            'matplotlib',
            'heartpy',
            'coverage',
            'pytest']


setup(
    name='Emo_Phys_Eval',
    version='0.2',
    packages=['Configs', 'Modules'],
    url='www.github.com/ZacDair/Emo_Phys_Eval',
    license='MIT',
    author='Zac Dair',
    author_email='zachary.dair@mycit.ie',
    description='Cardiac Signal Evaluation',
    install_requires=REQUIRES,
    python_requires=">=3.7.5"
)
