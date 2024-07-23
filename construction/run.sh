#/usr/bin/env sh

GO=/home/nishida/storage/dataset/GO/GO.csv
HOIP_ONTOLOGY=/home/nishida/storage/dataset/HOIP-Ontology/HOIP.csv
HOIP_FILE=COVID_ARDS_36006_lv2_231027
VERSION=v1


# Results:
#   - releases/VERSION/hoip_ontology.json
# python process_ontology.py \
#     --input_files ${GO} ${HOIP_ONTOLOGY} \
#     --output_file ../releases/${VERSION}/hoip_ontology.json
# 
# Results:
#   - releases/VERSION/HOIP_FILE.{train,dev,split}.json
python process_dataset.py \
    --input_file ./${HOIP_FILE}.csv \
    --ontology ../releases/${VERSION}/hoip_ontology.json \
    --output_dir ../releases/${VERSION}



