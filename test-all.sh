base_path=/home/jose_almeida/data/PROSTATEx/ProstateX_resized/
M=400
F=5
GAMMA=0.5

spatial_dim=$1

mkdir -p metrics_test

for mod in DWI T2WAx
do
    for C in gland # only gland annotations are available
    do
        # lower resolution of T2WAx in 3d, at least for now
        if [ $mod = "T2WAx" ] && [ $spatial_dim = "3d" ]
        then
            rate=0.5
        else 
            rate=1.0
        fi

        b="$mod.$C.$spatial_dim"
        best_fold=$(cat metrics/$b.csv | 
            grep iou | 
            awk -F, '{print $4","$2}' | 
            sort -nr | 
            head -1 | 
            cut -d ',' -f 2)
        python3 test-u-net-gland-decathlon.py \
            --root_dir /home/jose_almeida/data/ \
            --prostate_x_path $base_path \
            --metrics_path metrics_test/$b.csv \
            --config_file config/u-net-$spatial_dim.yaml \
            --checkpoint_path models/$mod.$C."$spatial_dim"_fold"$best_fold"_last.ckpt \
            --downsample_rate $rate \
            --dev 'cuda' \
            --mod $mod

        b="$mod.$C.$spatial_dim.augment"
        best_fold=$(cat metrics/$b.csv | 
            grep iou | 
            awk -F, '{print $4","$2}' | 
            sort -nr | 
            head -1 | 
            cut -d ',' -f 2)
        python3 test-u-net-gland-decathlon.py \
            --root_dir /home/jose_almeida/data/ \
            --prostate_x_path $base_path \
            --metrics_path metrics_test/$b.csv \
            --config_file config/u-net-$spatial_dim.yaml \
            --checkpoint_path models/$mod.$C.$spatial_dim.augment_fold"$best_fold"_last.ckpt \
            --downsample_rate $rate \
            --dev 'cuda' \
            --mod $mod
        done
done