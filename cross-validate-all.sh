base_path=/home/jose_almeida/data/PROSTATEx/ProstateX_resized/

for mod in DWI T2WAx
do
    for C in gland lesion gland_lesion
    do
        b="$mod.$C.2d"
        python3 train-u-net-prostate-x.py \
            --base_path $base_path \
            --classes $(echo gland_lesion | tr '_' ' ')\
            --config_file u-net-2d.yaml \
            --early_stopping 50 \
            --mod $mod \
            --seed 42 \
            --n_workers 8 \
            --loss_gamma 1.0 \
            --loss_comb 0.5 \
            --max_epochs 250 \
            --n_folds 5 \
            --checkpoint_dir models \
            --checkpoint_name $b \
            --summary_dir summaries \
            --summary_name $b \
            --metric_path metrics/$b.csv \
            --early_stopping 50

        b="$mod.$C.2d.augment"
        python3 train-u-net-prostate-x.py \
            --base_path $base_path \
            --classes $(echo gland_lesion | tr '_' ' ')\
            --config_file u-net-2d.yaml \
            --early_stopping 50 \
            --mod $mod \
            --seed 42 \
            --n_workers 8 \
            --loss_gamma 1.0 \
            --loss_comb 0.5 \
            --max_epochs 250 \
            --n_folds 5 \
            --checkpoint_dir models \
            --checkpoint_name $b \
            --summary_dir summaries \
            --summary_name $b \
            --metric_path metrics/$b.csv \
            --early_stopping 50 \
            --augment
    done
done

for mod in DWI T2WAx
do
    for C in gland lesion gland_lesion
    do
        b="$mod.$C.2d"
        python3 train-u-net-prostate-x.py \
            --base_path $base_path \
            --classes $(echo gland_lesion | tr '_' ' ')\
            --config_file u-net-3d.yaml \
            --early_stopping 50 \
            --mod $mod \
            --seed 42 \
            --n_workers 8 \
            --loss_gamma 1.0 \
            --loss_comb 0.5 \
            --max_epochs 250 \
            --n_folds 5 \
            --checkpoint_dir models \
            --checkpoint_name $b \
            --summary_dir summaries \
            --summary_name $b \
            --metric_path metrics/$b.csv \
            --early_stopping 50

        b="$mod.$C.2d.augment"
        python3 train-u-net-prostate-x.py \
            --base_path $base_path \
            --classes $(echo gland_lesion | tr '_' ' ')\
            --config_file u-net-3d.yaml \
            --early_stopping 50 \
            --mod $mod \
            --seed 42 \
            --n_workers 8 \
            --loss_gamma 1.0 \
            --loss_comb 0.5 \
            --max_epochs 250 \
            --n_folds 5 \
            --checkpoint_dir models \
            --checkpoint_name $b \
            --summary_dir summaries \
            --summary_name $b \
            --metric_path metrics/$b.csv \
            --early_stopping 50 \
            --augment
    done
done