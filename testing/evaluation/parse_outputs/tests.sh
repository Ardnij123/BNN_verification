#!/bin/bash
#PBS -N BNN_verification
#PBS -l select=1:ncpus=8:mem=16gb:scratch_local=2gb
#PBS -l walltime=4:00:00

THESIS=/storage/brno2/home/jindmen/bachelor_thesis
OUTPUTDIR=$THESIS/output/$PBS_JOBID
DATA=$THESIS/testing/clingo-based

mkdir -p "$OUTPUTDIR"
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR"

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set"; exit 1 ; }

cp -r -L $DATA $SCRATCHDIR/clingo-based
cd $SCRATCHDIR/clingo-based
module load clingo

EVALUATOR="./evaluator.py -t 8"
DOUBLESPACE='echo ""; echo ""'

eval $DOUBLESPACE
echo "TEST 1" ; >&2 echo "TEST 1"
# testing 100-50-20-10 on different inputs
# Hypothesis1: time to solve ~ number of models
# Hypothesis2: time to 1st model is equal for all
# Hypothesis3: aspif file size ~ does not change based on input instance
{
	INTER=intermediate.aspif
	TEST="$EVALUATOR -c hamming -d 3 -T 300 -x $INTER"
	MODELS=(
		[0]=models/mnist_bnn_1_blk_100_100_10/
		[1]=models/mnist_bnn_1_blk_100_50_10/
		[2]=models/mnist_bnn_2_blk_100_100_50_10/
		[3]=models/mnist_bnn_2_blk_100_50_20_10/
	)
	INPUTS=(
		[0]=inputs/instance_0_100.txt
		[1]=inputs/instance_1_100.txt
		[2]=inputs/instance_2_100.txt
		[3]=inputs/instance_3_100.txt
		[4]=inputs/instance_4_100.txt
		[5]=inputs/instance_5_100.txt
		[6]=inputs/instance_6_100.txt
		[7]=inputs/instance_7_100.txt
		[8]=inputs/instance_8_100.txt
		[9]=inputs/instance_9_100.txt
	)
	for mod in "${MODELS[@]}"; do
		for inp in "${INPUTS[@]}"; do
			eval $DOUBLESPACE
			echo "Model $mod on input $inp"
			eval $TEST -m $mod -i $inp | tail -n 5
			echo "Aspif stats: `wc $INTER`"
			rm $INTER
		done
	done
}

eval $DOUBLESPACE
echo "TEST 2" ; >&2 echo "TEST 2"
# testing 100-50-20-10 on different hamm dist
# Hypothesis1: time to first is dependent on hamm dist (how?)
# Hypothesis2: different hamm dists are similiar in size
{
	MODELS=(
	[0]="-m models/mnist_bnn_1_blk_100_50_10/ -i inputs/instance_0_100.txt"
	[1]="-m models/mnist_bnn_1_blk_400_100_10/ -i inputs/instance_0_400.txt"
	[2]="-m models/mnist_bnn_2_blk_100_50_20_10/ -i inputs/instance_0_100.txt"
	[3]="-m models/mnist_bnn_3_blk_25_25_25_20_10/ -i inputs/instance_0_25.txt"
	)
	INTER=intermediate.aspif
	TEST="$EVALUATOR -c hamming -T 300 -x $INTER"
	for mod in "${MODELS[@]}"; do
		for dist in 0 1 2 3 4; do
			eval $DOUBLESPACE
			echo "Model $mod on distance $dist"
			eval $TEST $mod -d $dist | tail -n 5
			echo "Aspif stats: `wc $INTER`"
			rm $INTER
		done
	done
}


eval $DOUBLESPACE
echo "TEST 3" ; >&2 echo "TEST 3"
# testing different models on different architectures
# Hypothesis: time to solve and aspif file size differs a lot based on architecture
{
	INTER=intermediate.aspif
	TEST="$EVALUATOR -c hamming -d 3 -T 300 -x $INTER"
	declare MODELS=(
		[0]='-m models/mnist_bnn_1_blk_64_10_10/ -i inputs/instance_0_64.txt'
		[1]='-m models/mnist_bnn_1_blk_100_100_10/ -i inputs/instance_0_100.txt'
		[2]='-m models/mnist_bnn_2_blk_36_15_10_10/ -i inputs/instance_0_36.txt'
		[3]='-m models/mnist_bnn_2_blk_100_50_20_10/ -i inputs/instance_0_100.txt'
	)
	declare ARCHITECTURES=(
		[0]='agg_2.lp'
		# TODO
	)
	for mod in "${MODELS[@]}"; do
		for arch in "${ARCHITECTURES[@]}"; do
			eval $DOUBLESPACE
			echo "Model $mod on architecture $arch"
			eval "$TEST $mod -s $arch" | tail -n 5
			echo "Aspif stats: `wc $INTER`"
			rm $INTER
		done
	done
}


eval $DOUBLESPACE
echo "TEST 4" ; >&2 echo "TEST 4"
# testing different inpbits
{
	MODELS=(
	[0]="-m models/mnist_bnn_1_blk_100_50_10/ -i inputs/instance_0_100.txt"
	[1]="-m models/mnist_bnn_1_blk_400_100_10/ -i inputs/instance_0_400.txt"
	[2]="-m models/mnist_bnn_2_blk_100_50_20_10/ -i inputs/instance_0_100.txt"
	[3]="-m models/mnist_bnn_3_blk_25_25_25_20_10/ -i inputs/instance_0_25.txt"
	)
	INTER=intermediate.aspif
	TEST="$EVALUATOR -c inpbits -T 300 -x $INTER"
	for mod in "${MODELS[@]}"; do
		inp="`echo $mod | cut -d'_' -f'5'`"
		for free in 0 8 16 24; do
			eval $DOUBLESPACE
			echo "Model $mod on $free free bits"
			eval $TEST $mod -f "inputs/inpbits_0_${inp}_${free}.txt" | tail -n 5
			echo "Aspif stats: `wc $INTER`"
			rm $INTER
		done
	done
}


clean_scratch
