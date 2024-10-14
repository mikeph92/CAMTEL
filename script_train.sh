# python train.py \
# --classification-task 'TIL' \
# --testset 'lizard'

# python train.py \
# --classification-task 'TIL' \
# --testset 'cptacCoad' 

# python train.py \
# --classification-task 'TIL' \
# --testset 'tcgaBrca'

# python train.py \
# --classification-task 'TIL' \
# --testset 'nucls' 


python train.py \
--classification-task 'tumor' \
--testset 'pannuke' 

python train.py \
--classification-task 'tumor' \
--testset 'ocelot' 

python train.py \
--classification-task 'tumor' \
--testset 'nucls' 


# # train for single headed tasks
# python train.py \
# --classification-task 'TIL' \
# --testset 'lizard' \
# --multitask '' 

# python train.py \
# --classification-task 'TIL' \
# --testset 'cptacCoad' \
# --multitask '' 

# python train.py \
# --classification-task 'TIL' \
# --testset 'tcgaBrca' \
# --multitask '' 

# python train.py \
# --classification-task 'TIL' \
# --testset 'nucls' \
# --multitask '' 

python train.py \
--classification-task 'tumor' \
--testset 'pannuke' \
--multitask ''

python train.py \
--classification-task 'tumor' \
--testset 'ocelot' \
--multitask ''

python train.py \
--classification-task 'tumor' \
--testset 'nucls' \
--multitask ''
