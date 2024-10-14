python train2.py \
--classification-task 'TIL' \
--testset 'lizard'

python train2.py \
--classification-task 'TIL' \
--testset 'cptacCoad' 

python train2.py \
--classification-task 'TIL' \
--testset 'tcgaBrca'

python train2.py \
--classification-task 'TIL' \
--testset 'nucls' 


# python train2.py \
# --classification-task 'tumor' \
# --testset 'pannuke' 

# python train2.py \
# --classification-task 'tumor' \
# --testset 'ocelot' 

# python train2.py \
# --classification-task 'tumor' \
# --testset 'nucls' 


# train2 for single headed tasks
python train2.py \
--classification-task 'TIL' \
--testset 'lizard' \
--multitask '' 

python train2.py \
--classification-task 'TIL' \
--testset 'cptacCoad' \
--multitask '' 

python train2.py \
--classification-task 'TIL' \
--testset 'tcgaBrca' \
--multitask '' 

python train2.py \
--classification-task 'TIL' \
--testset 'nucls' \
--multitask '' 

# python train2.py \
# --classification-task 'tumor' \
# --testset 'pannuke' \
# --multitask ''

# python train2.py \
# --classification-task 'tumor' \
# --testset 'ocelot' \
# --multitask ''

# python train2.py \
# --classification-task 'tumor' \
# --testset 'nucls' \
# --multitask ''
