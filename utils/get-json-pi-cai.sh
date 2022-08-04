source ./paths

mkdir -p dataset_information

echo python3 utils/generate-dataset-json.py \
    --input_path $PICAI_PATH_RESAMPLED_CORRECTED \
    --mask_path $PICAI_MASK_PATH_HUMAN/ \
    --class_csv_path $PICAI_MARKSHEET_ISUP \
    --mask_pattern "*/*nii.gz" \
    --patterns "*/*/*_t2w.mha" "*/*/*_adc.mha" "*/*/*_hbv.mha" \
    --output_json dataset_information/bb.lesion.pi-cai.json \
    --id_pattern "[0-9]+_[0-9]+" \
    --mask_key lesion_human

echo python3 utils/generate-dataset-json.py \
    --input_path $PICAI_PATH_RESAMPLED_CORRECTED \
    --mask_path $PICAI_MASK_PATH_AI/ \
    --class_csv_path $PICAI_MARKSHEET_ISUP \
    --mask_pattern "*/*nii.gz" \
    --patterns "*/*/*_t2w.mha" "*/*/*_adc.mha" "*/*/*_hbv.mha" \
    --output_json dataset_information/bb.lesion.pi-cai.ai.json \
    --id_pattern "[0-9]+_[0-9]+" \
    --mask_key lesion_ai

python3 utils/generate-dataset-json.py \
    --input_path $PICAI_PATH_RESAMPLED_CORRECTED \
    --mask_path $PICAI_MASK_PATH_GLAND_AI/ \
    --class_csv_path $PICAI_MARKSHEET_ISUP \
    --mask_pattern "*/*nii.gz" \
    --patterns "*/*/*_t2w.mha" "*/*/*_adc.mha" "*/*/*_hbv.mha" \
    --output_json dataset_information/bb.gland.pi-cai.ai.json \
    --id_pattern "[0-9]+_[0-9]+" \
    --mask_key gland