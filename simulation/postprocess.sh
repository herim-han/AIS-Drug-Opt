#!/bin/bash
#BASEDIR=/simulation
export PATH="$BASEDIR/mgltools_x86_64Linux2_1.5.6/bin:"$PATH
alias pythonsh="$BASEDIR/mgltools_x86_64Linux2_1.5.6/bin/pythonsh"

for f in $BASEDIR/output/*/ligand_*; do
        #if [ "$1" = "vina" ]; then
        #	 b=`basename $f`
        #else
        #        b=`basename $f`.pdbqt
        #fi

	b=`basename $f`
        
	#convert pdbqt to pdb
        cd $BASEDIR/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs
        pythonsh pdbqt_to_pdb.py -f ${f}/out.pdbqt -o ${f}/out.pdb
	if [ $? -ne 0 ]; then
		echo "Error in converting output file. (pdbqt to pdb)" >> $BASEDIR/output/log/error.log
		exit -1
	fi
	cd $BASEDIR
	obabel -i pdb ${f}/out.pdb -o pdb -O ${f}/out.pdb
	
	if [ $? -ne 0 ]; then
		echo "Error in converting pdb output file. (pdb to pdb with openbabel)" >> $BASEDIR/output/log/error.log
		exit -1
	fi

	#if [ ${b##*.} != "pdbqt" ]; then
	#	mv $BASEDIR/output/${b} $BASEDIR/output/${b}.pdbqt
	#fi
	
done

cd $BASEDIR
python3 parse_result.py
if [ $? -ne 0 ]; then
	echo "Error in parsing the result" >> $BASEDIR/output/log/error.log
	exit -1
fi

rm -f $BASEDIR/output/temp_img.png
