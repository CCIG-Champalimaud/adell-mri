base_path=/home/jose_almeida/data/PI-CAI/dataset_resampled_corrected/
mask_path=/home/jose_almeida/data/PI-CAI/labels/csPCa_lesion_delineations/human_expert/original/
mask_path_ai=/home/jose_almeida/data/PI-CAI/labels/csPCa_lesion_delineations/AI/
mask_path_gland_ai=/home/jose_almeida/data/PI-CAI/labels/anatomical_delineations/whole_gland/AI/
class_csv_path=/home/jose_almeida/data/PI-CAI/labels/clinical_information/marksheet_isup.csv

mkdir -p dataset_information

echo python3 utils/generate-dataset-json.py \
    --input_path $base_path \
    --mask_path $mask_path/ \
    --class_csv_path $class_csv_path \
    --mask_pattern "*/*nii.gz" \
    --patterns "*/*/*_t2w.mha" "*/*/*_adc.mha" "*/*/*_hbv.mha" \
    --output_json dataset_information/bb.lesion.pi-cai.json \
    --id_pattern "[0-9]+_[0-9]+" \
    --mask_key lesion_human

echo python3 utils/generate-dataset-json.py \
    --input_path $base_path \
    --mask_path $mask_path_ai/ \
    --class_csv_path $class_csv_path \
    --mask_pattern "*/*nii.gz" \
    --patterns "*/*/*_t2w.mha" "*/*/*_adc.mha" "*/*/*_hbv.mha" \
    --output_json dataset_information/bb.lesion.pi-cai.ai.json \
    --id_pattern "[0-9]+_[0-9]+" \
    --mask_key lesion_ai

python3 utils/generate-dataset-json.py \
    --input_path $base_path \
    --mask_path $mask_path_gland_ai/ \
    --class_csv_path $class_csv_path \
    --mask_pattern "*/*nii.gz" \
    --patterns "*/*/*_t2w.mha" "*/*/*_adc.mha" "*/*/*_hbv.mha" \
    --output_json dataset_information/bb.gland.pi-cai.ai.json \
    --id_pattern "[0-9]+_[0-9]+" \
    --mask_key gland