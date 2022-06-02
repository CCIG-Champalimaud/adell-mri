base_path=/home/jose_almeida/data/PROSTATEx/ProstateX_resized/
M=250
F=5
GAMMA=2.0

spatial_dim=$1

mkdir -p models
mkdir -p summaries
mkdir -p metrics
mkdir -p models/u-net-$spatial_dim
mkdir -p summaries/u-net-$spatial_dim
mkdir -p metrics/u-net-$spatial_dim

for mod in DWI T2WAx
do
    for C in gland lesion gland_lesion
    do
        # lower resolution of T2WAx in 3d, at least for now
        if [ $mod = "T2WAx" ] && [ $spatial_dim = "3d" ]
        then
            rate=0.5
        else 
            rate=1.0
        fi
        
        b="$mod.$C.$spatial_dim"
        python3 u-net-train-prostate-x.py \
            --base_path $base_path \
            --classes $(echo $C | tr '_' ' ')\
            --config_file config/u-net-$spatial_dim.yaml \
            --early_stopping 20 \
            --mod $mod \
            --seed 42 \
            --n_workers 8 \
            --loss_gamma $GAMMA \
            --loss_comb 0.5 \
            --max_epochs $M \
            --n_folds $F \
            --checkpoint_dir models/u-net-$spatial_dim \
            --checkpoint_name $b \
            --summary_dir summaries/u-net-$spatial_dim \
            --summary_name $b \
            --metric_path metrics/u-net-$spatial_dim/$b.csv \
            --dev cuda \
            --downsample_rate $rate

        b="$mod.$C.$spatial_dim.augment"
        python3 u-net-train-prostate-x.py \
            --base_path $base_path \
            --classes $(echo $C | tr '_' ' ')\
            --config_file config/u-net-$spatial_dim.yaml \
            --early_stopping 20 \
            --mod $mod \
            --seed 42 \
            --n_workers 8 \
            --loss_gamma $GAMMA \
            --loss_comb 0.5 \
            --max_epochs $M \
            --n_folds $F \
            --checkpoint_dir models/u-net-$spatial_dim \
            --checkpoint_name $b \
            --summary_dir summaries/u-net-$spatial_dim \
            --summary_name $b \
            --metric_path metrics/u-net-$spatial_dim/$b.csv \
            --dev cuda \
            --downsample_rate $rate \
            --augment
    done
done