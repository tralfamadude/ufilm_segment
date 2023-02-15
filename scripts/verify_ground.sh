#!/bin/bash
# arg is path to combined/ground.csv
#  This will verify that combined/images/* files are in ground.csv
GROUND=$1
DATA_DIR=$(dirname $GROUND)
echo " verifying $DATA_DIR"
BASENAMES_I=$(cd $DATA_DIR/images; ls -1 * | awk -F. '{ print $1 }')
BASENAMES_L=$(cd $DATA_DIR/labels; ls -1 * | awk -F. '{ print $1 }')

for b in $BASENAMES_I ; do
  if grep -q $b $GROUND ; then 
    :
  else
    echo "Missing: $b corresponding to combined/images file"
  fi
done

for b in $BASENAMES_L ; do
  if grep -q $b $GROUND ; then
    :
  else
    echo "Missing: $b corresponding to combined/labels file"
  fi
done
