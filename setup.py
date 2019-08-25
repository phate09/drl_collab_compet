from distutils.core import setup

from Cython.Build import cythonize

# setup(ext_modules=cythonize('ilqr_server2.pyx'))

setup(
    ext_modules=cythonize(["utility/PrioritisedExperienceReplayBuffer_cython.pyx",
                           ], language_level="3")
)
