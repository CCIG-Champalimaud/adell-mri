base_path=/home/jose_almeida/data/PROSTATEx/ProstateX_resized
DEV=cuda

spatial_dim=$1
dataset=$2

mkdir -p predictions
mkdir -p predictions/prostate-x

for mod in DWI T2WAx
do
    for C in gland lesion
    do
        # lower resolution of T2WAx in 3d, at least for now
        if [ $mod = "T2WAx" ] && [ $spatial_dim = "3d" ]
        then
            rate=0.5
        else 
            rate=1.0
        fi

        if [ $mod = "T2WAx" ]
        then 
            index=0
        else
            index=1
        fi

        b="$mod.$C.$spatial_dim"
        best_fold=$(cat metrics/u-net-$spatial_dim/$b.csv | 
            grep iou | 
            awk -F, '{print $4","$2}' | 
            sort -nr | 
            head -1 | 
            cut -d ',' -f 2)
        val_ids=$(cat metrics/u-net-$spatial_dim/$b.csv | 
            grep val_ids | 
            grep val_ids,$best_fold | 
            cut -d ',' -f 4)
        scan_paths=$(find $base_path/$mod/ -name "*nii.gz" |
            grep -E "$(echo $val_ids | tr ':' '|')" | 
            xargs)
        python3 u-net-predict.py \
            --input_path $scan_paths \
            --mod $mod \
            --prostate_x_path $base_path \
            --output_path predictions/prostate-x/$b \
            --config_file config/u-net-$spatial_dim.yaml \
            --checkpoint_path models/u-net-$spatial_dim/$mod.$C."$spatial_dim"_fold"$best_fold"_last.ckpt \
            --n_workers 8 \
            --downsample_rate $rate \
            --dev $DEV

        b="$mod.$C.$spatial_dim.augment"
        best_fold=$(cat metrics/u-net-$spatial_dim/$b.csv | 
            grep iou | 
            awk -F, '{print $4","$2}' | 
            sort -nr | 
            head -1 | 
            cut -d ',' -f 2)
        val_ids=$(cat metrics/u-net-$spatial_dim/$b.csv | 
            grep val_ids | 
            grep val_ids,$best_fold | 
            cut -d ',' -f 4)
        scan_paths=$(find $base_path/$mod/ -name "*nii.gz" |
            grep -E "$(echo $val_ids | tr ':' '|')" | 
            xargs)
        python3 u-net-predict.py \
            --input_path $scan_paths \
            --mod $mod \
            --prostate_x_path $base_path \
            --output_path predictions/prostate-x/$b \
            --config_file config/u-net-$spatial_dim.yaml \
            --checkpoint_path models/u-net-$spatial_dim/$mod.$C.$spatial_dim.augment_fold"$best_fold"_last.ckpt \
            --n_workers 8 \
            --downsample_rate $rate \
            --dev $DEV
    done
done