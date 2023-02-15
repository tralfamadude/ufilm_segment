#!/bin/bash
# usage:  top_dest_base_dir  issue_id

BASE_DIR=$1
ISSUE_ID=$2

mkdir -p $BASE_DIR/$ISSUE_ID
cd $BASE_DIR

# is it already downloaded?
if [ -d $ISSUE_ID ]; then
  echo "Already downloaded "
  exit 0
fi

HOCR=$(ia list $ISSUE_ID | grep hocr.html.gz | head -1)
if [ ! -z "$HOCR" ] ; then
  ia download $ISSUE_ID $HOCR
else
  echo "No hocr file for $ISSUE_ID ; no action taken"
  exit 0
fi

ia download $ISSUE_ID ${ISSUE_ID}_jp2.zip

cd ${ISSUE_ID}

unzip ${ISSUE_ID}_jp2.zip >/dev/null
gunzip $HOCR
mkdir -p images

for j in *_jp2/*.jp2 ; do 
  opj_decompress -r 1 -i $j -o images/$(basename $j .jp2).png >/dev/null 2>/dev/null
done

rm ${ISSUE_ID}_jp2.zip
rm -rf *_jp2
