mkdir -p dataset_information

iou=0.4
for file in dataset_information/bb*
do  
    echo $file
    suffix=$(echo $file | cut -d '/' -f 2 | cut -d '.' -f 2-100)
    python3 utils/bounding-boxes-to-anchors.py \
        --input_path $file \
        --spatial_dim 3 \
        --iou_threshold $iou > dataset_information/anchors.$suffix
done
