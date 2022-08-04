base_path=/home/jose_almeida/data/PROSTATEx/ProstateX

mkdir -p dataset_information

for region in gland lesion
do
    for mod in T2WAx DWI
    do
        if [ "$mod" = DWI ]
        then
            patterns="ADC/*nii.gz DWI/*nii.gz"
            mask_pattern="*DWI*nii.gz"
        else
            patterns="$mod/*1.nii.gz"
            mask_pattern="*$mod*nii.gz"
        fi
        
        echo $region $mod
        python3 utils/generate-dataset-json.py \
            --input_path $base_path \
            --mask_path $base_path/aggregated-labels-$region/ \
            --mask_pattern $mask_pattern \
            --patterns $patterns \
            --output_json dataset_information/bb.$region.$mod.prostate_x.json \
            --id_pattern "Prostatex[0-9]+"
    done
done