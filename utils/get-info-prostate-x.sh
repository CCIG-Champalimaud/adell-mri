source utils/paths

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
            --input_dir $PROSTATE_X_RESIZED_PATH/$mod \
            --parameter $info \
            --quantile $q > dataset_information/$info.$mod
    done
done