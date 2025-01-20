#! /bin/bash
#BASEDIR=/simulation
export PATH="$BASEDIR/mgltools_x86_64Linux2_1.5.6/bin:"$PATH
alias pythonsh='$BASEDIR/mgltools_x86_64Linux2_1.5.6/bin/pythonsh'

LOGDIR=$BASEDIR/output/log
mkdir -p $BASEDIR/output/log

python3 $BASEDIR/create_pdb.py
if [ $? -ne 0 ]; then
	echo "Error in creating input pdb file. (Invalid ligand files or receptor files)" >> ${LOGDIR}/error.log
	exit -1
fi

cd $BASEDIR/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs

mkdir -p $BASEDIR/input/grid
mkdir -p $BASEDIR/input/config
mkdir -p $BASEDIR/input/converted_receptor
mkdir -p $BASEDIR/input/converted_ligand
#mkdir -p /simulation/output/log
#convert receptor file
for recp in $BASEDIR/input/receptor/*; do
	recp_b=`basename $recp`
	recp_name="${recp_b%.*}"
	pythonsh prepare_receptor4.py -r $recp -o $BASEDIR/input/converted_receptor/$recp_name.pdbqt
	if [ $? -ne 0 ]; then
		echo "Error in converting a input receptor. ($recp_b to pdbqt)" >> ${LOGDIR}/error.log
		exit -1
	fi
	mkdir -p $BASEDIR/input/grid/$recp_name

	if [ ! -f $BASEDIR/input/user.log ]; then
		$BASEDIR/ghecom -ipdb $BASEDIR/input/converted_receptor/$recp_name.pdbqt -opocpdb $BASEDIR/input/grid/$recp_name/out.pocket
	#if [ $? -ne 0 ]; then
	#	echo "Error in calculating pocket" >> ${LOGDIR}/error.log
	#	exit -1
	#fi
	fi

	if [ ! -f $BASEDIR/input/grid/$recp_name/out.pocket ]; then
		echo "Error in calculating a pocket of $recp_b" >> ${LOGDIR}/error.log
		exit -1
	fi
done

for f in $BASEDIR/input/ligand/*; do	
	b=`basename $f`
	
	if [ "$1" = "vina" ]; then
		cd $BASEDIR/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs
		#convert ligand file
		pythonsh prepare_ligand4.py -l ${f} -o $BASEDIR/input/converted_ligand/$b.pdbqt
		if [ $? -ne 0 ]; then
			echo "Error in converting a ligand file. ($b to pdbqt)" >> ${LOGDIR}/error.log
			exit -1
		fi
		#mkdir -p /simulation/output/$b.pdbqt
		#mkdir -p /simulation/input/config/$b.pdbqt
	else
		ext="${b##*.}"
		if [ $ext = "pdb" ] || [ $ext = "pdbqt" ]; then
			cd $BASEDIR/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs
                	#convert ligand file
                	pythonsh prepare_ligand4.py -l ${f} -o $BASEDIR/input/converted_ligand/$b.pdbqt
                	if [ $? -ne 0 ]; then
				echo "Error in converting a ligand file. ($b to pdbqt)" >> ${LOGDIR}/error.log
                        	exit -1
                	fi
			#mkdir -p $BASEDIR/output/$b.pdbqt
			#mkdir -p /simulation/input/config/$b.pdbqt
		else
			cp $f $BASEDIR/input/converted_ligand/
			#mkdir -p /simulation/output/$b
			#mkdir -p /simulation/input/config/$b
		fi
	fi

	#mkdir -p /simulation/output/$b.pdbqt
	#mkdir -p /simulation/input/config/$b.pdbqt
	#prepare grid parameter file
	#if [ ! -f /simulation/input/user.log ]; then
		
		#pythonsh prepare_gpf4.py -l /simulation/input/converted_ligand/$b.pdbqt -r /simulation/input/receptor.pdbqt -o /simulation/input/grid/receptor.gpf
		#if [ $? -ne 0 ]; then
		#	echo "Error in preparing gpf" >> ${LOGDIR}/error.log
		#	exit -1
		#fi

		#calculate the pocket size
		#cp /simulation/input/receptor.pdbqt /simulation/input/grid/receptor.pdbqt
		#cd /simulation/input/grid
		#autogrid4 -p receptor.gpf
		#if [ $? -ne 0 ]; then
		#	echo "Error in calculating pocket size" >> ${LOGDIR}/error.log
		#	exit -1
		#fi
		

		#cp -r /simulation/input/grid /simulation/output/${b}.pdbqt/
	#fi
	
	#mkdir -p /simulation/output/$b.pdbqt
	#write config file
	#python3 /simulation/create_conf.py --path /simulation/input/config/$b.pdbqt
	#if [ $? -ne 0 ]; then
	#	echo "Error in creating config file" >> ${LOGDIR}/error.log
	#	exit -1
	#fi

	#rm -f /simulation/input/grid/*
done

python3 $BASEDIR/create_conf.py
if [ $? -ne 0 ]; then
	echo "Error in creating config file for simulation" >> ${LOGDIR}/error.log
	exit -1
fi
#ls > /simulation/input/ligandlist /simulation/input/converted_ligand
#mkdir -p /simulation/input/ProcessedLigand

#ls > ligandlist /simulation/input/converted_ligand
#mkdir -p /simulation/input/ProcessedLigand

#/usr/local/bin/mpirun -np 5 /simulation/mpiVINA > /simulation/output/mpivina.log

#run autodock vina
#/simulation/autodock_vina_1_1_2_linux_x86/bin/vina --config /simulation/input/conf.txt --ligand /simulation/input/ligand.pdbqt --out /simulation/output/${b}/out.pdbqt --log /simulation/output/${b}/log.txt

#for f in /simulation/input/ProcessedLigand/*; do
#	b=`basename $f`
	#convert pdbqt to pdb
#	cd /simulation/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs
#	pythonsh pdbqt_to_pdb.py -f /simulation/output/${b}/out.pdbqt -o /simulation/output/${b}/out.pdb
