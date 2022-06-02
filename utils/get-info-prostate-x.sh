base_path=/home/jose_almeida/data/PROSTATEx/ProstateX_resized

mkdir -p dataset_information

for info in spacing size
do
    for mod in T2WAx DWI
    do
        if [ $info = spacing ]
        then
            q=0
        else
            q=0.5
        fi

        python3 utils/get-info.py \
            --input_dir $base_path/$mod \
            --parameter $info \
            --quantile $q > dataset_information/$info.$mod
    done
done