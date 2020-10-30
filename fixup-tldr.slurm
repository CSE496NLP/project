#!/bin/sh
#SBATCH --time=1-12:00:00
#SBATCH --mem=64G
#SBATCH --job-name=get-ppdb-tldr-lhs
#SBATCH --error=ppdb-tldr-lhs.err
#SBATCH --output=ppdb-tldr-lhs.out
#SBATCH --licenses=common

OUTDIR=/work/scott/swflint/ppdb-data/
PPDBFILEBASE=ppdb-tldr

mv $OUTDIR/${PPDBFILEBASE}-phrases $OUTDIR/${PPDBFILEBASE}-phrases.bak
mv $OUTDIR/${PPDBFILEBASE}-paraphrases $OUTDIR/${PPDBFILEBASE}-paraphrases.bak


cat ${OUTDIR}/${PPDBFILEBASE}-phrases.bak | sed -e 's/\\//' -e 's/xc[[:digit:]]*//' -e 's/\\//' -e 's/xc2//' | awk '$1=$1' > ${OUTDIR}/${PPDBFILEBASE}-phrases
cat ${OUTDIR}/${PPDBFILEBASE}-paraphrases.bak | sed -e 's/\\//' -e 's/xc[[:digit:]]*//' -e 's/\\//' -e 's/xc2//' | awk '$1=$1' > ${OUTDIR}/${PPDBFILEBASE}-paraphrases
