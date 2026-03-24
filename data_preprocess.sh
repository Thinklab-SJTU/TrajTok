set -e

INPUT_DIR=Your_Waymo_Open_Dataset_Path    # .../waymo_open_dataset_motion_v_1_3_0/scenario/
OUTPUT_DIR=./data/waymo_processed
NUM_WORKERS=12

python data_preprocess.py --split training --num_workers $NUM_WORKERS --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR
python data_preprocess.py --split validation --num_workers $NUM_WORKERS --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR
python data_preprocess.py --split testing --num_workers $NUM_WORKERS --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR