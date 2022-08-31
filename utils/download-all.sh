L="GLAND.Tabular.nested.50F.new
   CROP_GLAND.Tabular.nested.50F.new
   TZ.Tabular.nested.50F 
   CROP_TZ.Tabular.nested.50F 
   CROP_PZ.Tabular.nested.50F 
   PZ.Tabular.nested.50F
   GLAND.Tabular.nested.AEC.50F
   GLAND.Tabular.nested.50F.monica
   CROP_GLAND.Tabular.nested.50F.monica
   PZ.Tabular.nested.50F.monica
   TZ.Tabular.nested.50F.monica"

o=performance_data
mkdir -p $o

for file in $L
do
    file_nd=$(echo $file | tr '.' ' ')
    python3 scripts/download_wandb_report.py --input_path "ccig/$file_nd" --output_path $o/$file.csv
done