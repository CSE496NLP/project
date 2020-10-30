#!/bin/sh
#SBATCH --time=1-12:00:00
#SBATCH --mem=64G
#SBATCH --job-name=get-ppdb-tldr-lhs
#SBATCH --error=ppdb-tldr-phrases.err
#SBATCH --output=ppdb-tldr-phrases.out
#SBATCH --licenses=common

WIKIDIR=/common/cse896/asturtz/data-simplification/wikilarge/
OUTDIR=/work/scott/swflint/ppdb-data/

echo Total Source Sentences:
SRC_SENT=$(cat $WIKIDIR/*.train.src | wc -l)
echo $SRC_SENT

echo Number of sentences Containing "Phrases":
cat /work/scott/swflint/wikilarge/* | grep -Fi -f $OUTDIR/ppdb-tldr-phrases | wc -l
# echo $NUM_CONT

# PERCENT=$(( NUM_CONT * 100 / SRC_SENT ))

# echo Percent phrases used: ${PERCENT}%
