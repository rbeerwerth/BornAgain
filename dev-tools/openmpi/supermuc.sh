# Setup for BornAgain compilation at SuperMUC LRZ cluster


module load git
#module load gcc/4.5

module load gcc/4.7
module load cmake/2.8
#module load boost/1.47_gcc
module load fftw/mpi/3.3
module load gsl

#export FFTW3=/lrz/sys/libraries/fftw/3.3.3/avx/
export FFTW3=$FFTW_BASE
export GSL_DIR=$GSL_BASE
#export BOOST_ROOT=$BOOST_BASE
export BOOST_ROOT=/home/hpc/pr87me/di29sok/software/boost_1_55_0.gcc47


