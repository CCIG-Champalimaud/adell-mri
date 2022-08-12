mkdir -p dataset_information

source utils/paths

python3 utils/generate-dataset-json.py \
    --input_path $PICAI_PATH_CORRECTED \
    --mask_path $PICAI_MASK_PATH_HUMAN \
    --class_csv_path $PICAI_MARKSHEET_ISUP \
    --mask_pattern "*/*nii.gz" \
    --patterns "*/*/*_t2w.mha" "*/*/*_adc.mha" "*/*/*_hbv.mha" \
    --output_json dataset_information/bb.lesion.pi-cai.json \
    --id_pattern "[0-9]+_[0-9]+" \
    --mask_key lesion_human

python3 utils/generate-dataset-json.py \
    --input_path $PICAI_PATH_CORRECTED \
    --mask_path $PICAI_MASK_PATH_AI \
    --class_csv_path $PICAI_MARKSHEET_ISUP \
    --mask_pattern "*/*nii.gz" \
    --patterns "*/*/*_t2w.mha" "*/*/*_adc.mha" "*/*/*_hbv.mha" \
    --output_json dataset_information/bb.lesion.pi-cai.ai.json \
    --id_pattern "[0-9]+_[0-9]+" \
    --mask_key lesion_ai

python3 utils/generate-dataset-json.py \
    --input_path $PICAI_PATH_CORRECTED \
    --mask_path $PICAI_MASK_PATH_GLAND_AI \
    --class_csv_path $PICAI_MARKSHEET_ISUP \
    --mask_pattern "*/*nii.gz" \
    --patterns "*/*/*_t2w.mha" "*/*/*_adc.mha" "*/*/*_hbv.mha" \
    --output_json dataset_information/bb.gland.pi-cai.ai.json \
    --id_pattern "[0-9]+_[0-9]+" \
    --mask_key gland

python3 utils/generate-dataset-json.py \
    --input_path $PICAI_PATH_CORRECTED \
    --mask_path $PICAI_MASK_PATH_MERGE \
    --class_csv_path $PICAI_MARKSHEET_ISUP \
    --mask_pattern "*nii.gz" \
    --patterns "*/*/*_t2w.mha" "*/*/*_adc.mha" "*/*/*_hbv.mha" \
    --output_json dataset_information/bb.lesion.pi-cai.merged.json \
    --id_pattern "[0-9]+_[0-9]+" \
    --mask_key lesion_merge

python3 utils/merge-json-files.py \
    --input_paths \
    dataset_information/bb.lesion.pi-cai.merged.json \
    dataset_information/bb.lesion.pi-cai.ai.json\
    dataset_information/bb.gland.pi-cai.ai.json \
    dataset_information/bb.lesion.pi-cai.json \
    --suffixes lesion_human lesion_ai lesion_merge gland_ai \
    --rename image_labels_merge,image_labels > dataset_information/bb.pi-cai.json

python3 utils/remove-constant-masks.py \
    --input_json dataset_information/bb.pi-cai.json \
    --mask_keys lesion_merge > dataset_information/bb.pi-cai.nc.json