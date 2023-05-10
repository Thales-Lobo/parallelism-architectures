#PBS -S /bin/bash
#PBS -N tp1_ex2_sortie_de_cache
#PBS -e errorJob.txt
#PBS -j oe
#PBS -l walltime=0:04:00
#PBS -l select=1:ncpus=20:cpugen=skylake
#PBS -l place=excl
#PBS -m abe -M laurent.cabaret@centralesupelec.fr
#PBS -P progpar


# Load the same modules as for compilation
module load gcc/7.3.0
# module load intel-compilers/2019.3
# Go to the current directory
cd $PBS_O_WORKDIR


echo "Avec vectorisation"
for I in {32..4096..16}
do
(( ITER=10))
exe/tp1_ex2_vec.exe $ITER $I
done

echo "Sans vectorisation"
for I in {32..4096..16}
do
(( ITER=10))
exe/tp1_ex2_novec.exe $ITER $I
done

/gpfs/opt/bin/fusion-whereami
date
time sleep 2