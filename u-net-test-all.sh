base_path=/home/jose_almeida/data/PROSTATEx/ProstateX_resized/
DEV=cpu

spatial_dim=$1

mkdir -p metrics_test
mkdir -p metrics_test/u-net-$spatial_dim

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
        best_fold=$(cat metrics/u-net-$spatial_dim/$b.csv | 
            grep iou | 
            awk -F, '{print $4","$2}' | 
            sort -nr | 
            head -1 | 
            cut -d ',' -f 2)
        python3 u-net-test-gland-decathlon.py \
            --root_dir /home/jose_almeida/data/ \
            --prostate_x_path $base_path \
            --metrics_path metrics_test/u-net-$spatial_dim/$b.csv \
            --config_file config/u-net-$spatial_dim.yaml \
            --checkpoint_path models/u-net-$spatial_dim/$mod.$C."$spatial_dim"_fold"$best_fold"_last.ckpt \
            --downsample_rate $rate \
            --dev $DEV \
            --mod $mod

        b="$mod.$C.$spatial_dim.augment"
        best_fold=$(cat metrics/u-net-$spatial_dim/$b.csv | 
            grep iou | 
            awk -F, '{print $4","$2}' | 
            sort -nr | 
            head -1 | 
            cut -d ',' -f 2)
        python3 u-net-test-gland-decathlon.py \
            --root_dir /home/jose_almeida/data/ \
            --prostate_x_path $base_path \
            --metrics_path metrics_test/u-net-$spatial_dim/$b.csv \
            --config_file config/u-net-$spatial_dim.yaml \
            --checkpoint_path models/u-net-$spatial_dim/$mod.$C.$spatial_dim.augment_fold"$best_fold"_last.ckpt \
            --downsample_rate $rate \
            --dev $DEV \
            --mod $mod
        done
done