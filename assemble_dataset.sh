#!/bin/bash
# Assemble training data sets from annotated journal issues.

SRC_BASE=$1
DEST_DIR=$2
ISSUE_ID_TRAINING_LIST="$3"
ISSUE_ID_EVAL_LIST="$4"
ISSUE_ID_WITHHELD_LIST="$5"

function print_usage() {
    echo "Usage:  SrcBase DestDir  issue_id1,issue_id2,issue_id3  issue_id4,issue_id5  issue_id6,issue_id7";
    echo "  where first issue list is for training, second is for eval." 
    echo "   third list is for withheld."
    echo "   DestDir will be created."
    echo "Example: $0 ~/cvat_post ~/ufilm_datasets/pub_journal-of-thought_v1  sim_journal-of-thought_2004_spring_39_1,sim_journal-of-thought_2004_summer_39_2  sim_journal-of-thought_2004_winter_39_3,sim_journal-of-thought_2004_winter_39_4  sim_journal-of-thought_2005_fall_40_3";
    exit 1
}


# validate
if [ ! -d "$SRC_BASE" ] ; then
    print_usage
fi
if [ -z "$ISSUE_ID_TRAINING_LIST" ] ; then
    print_usage
fi
if [ -z "$ISSUE_ID_EVAL_LIST" ] ; then
    print_usage
fi
if [ -e "$DEST_DIR" ] ; then
    echo "no action taken, directory already exists: $DEST_DIR"
    exit 2
fi

function combine_stats_txt() {
    local dtype="$1"
    shift
    local in_file_list="$*"
    # tally stats.txt
    _task_name=""
    _image_count=0
    _image_annotated_count=0
    _annotation_count=0
    _title_article=0
    _authors_article=0
    _refs_article=0
    _toc=0
    for j in $in_file_list ; do
        # we source the stats.txt file to get tally values, but first line needs different treatment
        sfile=/tmp/tmp$$.txt
        cat $j | grep -v "^task_name=" >$sfile
        local task_names="$(cat $j | grep "^task_name=" | awk -F= '{ print $2 }')"
        . $sfile
        _task_name="$_task_name $task_names"
        let _image_count=_image_count+image_count
        let _image_annotated_count=_image_annotated_count+image_annotated_count
        let _annotation_count=_annotation_count+annotation_count
        let _title_article=_title_article+title_article
        let _authors_article=_authors_article+authors_article
        let _refs_article=_refs_article+refs_article
        let _toc=_toc+toc
    done
    rm /tmp/tmp$$.txt

    echo "task_name=$_task_name" >>$DEST_DIR/$dtype/stats.txt
    echo "image_count=$_image_count" >>$DEST_DIR/$dtype/stats.txt
    echo "image_annotated_count=$_image_annotated_count" >>$DEST_DIR/$dtype/stats.txt
    echo "annotation_count=$_annotation_count" >>$DEST_DIR/$dtype/stats.txt
    echo "title_article=$_title_article" >>$DEST_DIR/$dtype/stats.txt
    echo "authors_article=$_authors_article" >>$DEST_DIR/$dtype/stats.txt
    echo "refs_article=$_refs_article" >>$DEST_DIR/$dtype/stats.txt
    echo "toc=$_toc" >>$DEST_DIR/$dtype/stats.txt

}

function create_dataset_type() {
    local TYPE=$1
    local ISSUE_ID_LIST=$2

    # split issue list into space separated tokems
    ISSUE_ID_LIST=$(echo $ISSUE_ID_LIST | tr ',' ' ')

    mkdir -p $DEST_DIR/$TYPE/images
    mkdir -p $DEST_DIR/$TYPE/labels

    # copy the images and label images
    for j in $ISSUE_ID_LIST ; do
        cp $SRC_BASE/$j/images/* $DEST_DIR/$TYPE/images
        cp $SRC_BASE/$j/labels/* $DEST_DIR/$TYPE/labels
    done

    # assemble combined ground.csv and stats.csv
    for j in $ISSUE_ID_LIST ; do
        cat $SRC_BASE/$j/ground.csv | head -1 >$DEST_DIR/$TYPE/ground.csv.header
        cat $SRC_BASE/$j/ground.csv | grep -v file_basename >>$DEST_DIR/$TYPE/ground.csv.body
        cat $SRC_BASE/$j/stats.csv | head -1 >$DEST_DIR/$TYPE/stats.csv.header
        cat $SRC_BASE/$j/stats.csv | grep -v issue_id >>$DEST_DIR/$TYPE/stats.csv.body
        cp $SRC_BASE/$j/classes.txt $DEST_DIR
    done
    cat $DEST_DIR/$TYPE/ground.csv.header $DEST_DIR/$TYPE/ground.csv.body > $DEST_DIR/$TYPE/ground.csv
    cat $DEST_DIR/$TYPE/stats.csv.header $DEST_DIR/$TYPE/stats.csv.body > $DEST_DIR/$TYPE/stats.csv
    rm $DEST_DIR/$TYPE/*.csv.header $DEST_DIR/$TYPE/*.csv.body 

    # tally stats.txt
    local file_list=""
    for j in $ISSUE_ID_LIST ; do
        file_list="${file_list} ${SRC_BASE}/$j/stats.txt"
    done
    combine_stats_txt $TYPE $file_list

}

create_dataset_type training $ISSUE_ID_TRAINING_LIST
create_dataset_type eval $ISSUE_ID_EVAL_LIST
create_dataset_type withheld $ISSUE_ID_WITHHELD_LIST


# create config file for dhSegment training 
#
echo "{
  \"training_params\" : {
      \"learning_rate\": 5e-5,
      \"batch_size\": 1,
      \"make_patches\": false,
      \"training_margin\" : 0,
      \"n_epochs\": 30,
      \"data_augmentation\" : true,
      \"data_augmentation_max_rotation\" : 0.02,
      \"data_augmentation_max_scaling\" : 0.02,
      \"data_augmentation_flip_lr\": true,
      \"data_augmentation_flip_ud\": false,
      \"data_augmentation_color\": false,
      \"evaluate_every_epoch\" : 10
  },
  \"pretrained_model_name\" : \"resnet50\",
  \"prediction_type\": \"CLASSIFICATION\",
  \"train_data\" : \"$DEST_DIR/training\",
  \"eval_data\" : \"$DEST_DIR/eval\",
  \"classes_file\" : \"$DEST_DIR/classes.txt\",
  \"model_output_dir\" : \"$DEST_DIR/output\",
  \"gpu\" : \"0\"
}" > $DEST_DIR/ufilm_config.json

# ToDo: make combined dir for post-model training which uses both training and eval set against combined ground.csv
mkdir -p $DEST_DIR/combined/images
mkdir -p $DEST_DIR/combined/labels

for j in $DEST_DIR/training/images/* ; do 
    ln $j $DEST_DIR/combined/images/$(basename $j)
done

for j in $DEST_DIR/eval/images/* ; do 
    ln $j $DEST_DIR/combined/images/$(basename $j)
done

for j in $DEST_DIR/training/labels/* ; do 
    ln $j $DEST_DIR/combined/labels/$(basename $j)
done

for j in $DEST_DIR/eval/labels/* ; do 
    ln $j $DEST_DIR/combined/labels/$(basename $j)
done

# make  combined/ground.csv 
for j in $DEST_DIR/training/ground.csv $DEST_DIR/eval/ground.csv ; do
    cat $j | head -1 >$DEST_DIR/combined/ground.csv.header
    cat $j | grep -v file_basename >>$DEST_DIR/combined/ground.csv.body
done
cat $DEST_DIR/combined/ground.csv.header $DEST_DIR/combined/ground.csv.body > $DEST_DIR/combined/ground.csv
rm $DEST_DIR/combined/*.csv.header $DEST_DIR/combined/*.csv.body 

# make combined/stats.csv
for j in $DEST_DIR/training/stats.csv $DEST_DIR/eval/stats.csv ; do
    cat $j | head -1 >$DEST_DIR/combined/stats.csv.header
    cat $j | grep -v file_basename >>$DEST_DIR/combined/stats.csv.body
done
cat $DEST_DIR/combined/stats.csv.header $DEST_DIR/combined/stats.csv.body > $DEST_DIR/combined/stats.csv
rm $DEST_DIR/combined/*.csv.header $DEST_DIR/combined/*.csv.body 

# make combined stats.txt 
combine_stats_txt combined $DEST_DIR/training/stats.txt  $DEST_DIR/eval/stats.txt

