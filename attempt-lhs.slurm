#!/bin/sh
#SBATCH --time=1-12:00:00
#SBATCH --mem=64G
#SBATCH --job-name=get-ppdb-tldr-lhs
#SBATCH --error=ppdb-tldr-lhs.err
#SBATCH --output=ppdb-tldr-lhs.out
#SBATCH --licenses=common

PPDBDIR=/common/cse896/asturtz/data-simplification/ppdb/
OUTDIR=/work/scott/swflint/ppdb-data/
PPDBFILEBASE=ppdb-tldr

mkdir -p $OUTDIR

cat $PPDBDIR/$PPDBFILEBASE | awk -F ' \\|\\|\\| ' '{print $2}' > $OUTDIR/${PPDBFILEBASE}-phrases
cat $PPDBDIR/$PPDBFILEBASE | awk -F ' \\|\\|\\| ' '{print $3}' > $OUTDIR/${PPDBFILEBASE}-paraphrases
