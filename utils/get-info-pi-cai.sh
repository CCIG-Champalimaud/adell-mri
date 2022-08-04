python3 utils/get-info.py \
	--input_dir /home/jose_almeida/data/PI-CAI/dataset/ \
	--pattern "*/*/*_adc.mha" \
	--parameter spacing \
	--quantile 0.9 > dataset_information/spacing.ADC.PICAI

python3 utils/get-info.py \
	--input_dir /home/jose_almeida/data/PI-CAI/dataset/ \
	--pattern "*/*/*_hbv.mha" \
	--parameter spacing \
	--quantile 0.9 > dataset_information/spacing.HBV.PICAI

python3 utils/get-info.py \
	--input_dir /home/jose_almeida/data/PI-CAI/dataset/ \
	--pattern "*/*/*_t2w.mha" \
	--parameter spacing \
	--quantile 0.9 > dataset_information/spacing.T2WAx.PICAI

python3 utils/get-info.py \
	--input_dir /home/jose_almeida/data/PI-CAI/dataset/ \
	--pattern "*/*/*_adc.mha" \
	--parameter size \
	--quantile 0.5 > dataset_information/size.ADC.PICAI

python3 utils/get-info.py \
	--input_dir /home/jose_almeida/data/PI-CAI/dataset/ \
	--pattern "*/*/*_hbv.mha" \
	--parameter size \
	--quantile 0.5 > dataset_information/size.HBV.PICAI

python3 utils/get-info.py \
	--input_dir /home/jose_almeida/data/PI-CAI/dataset/ \
	--pattern "*/*/*_t2w.mha" \
	--parameter size \
	--quantile 0.5 > dataset_information/size.T2WAx.PICAI