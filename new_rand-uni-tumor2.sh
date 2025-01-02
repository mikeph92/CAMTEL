######################################

# python train_rand.py \
# --classification-task 'tumor' \
# --model 'UNI' \
# --crop-size 96 \
# --testset 'ocelot' 

python test_rand.py \
--classification-task 'tumor' \
--model 'UNI' \
--crop-size 96 \
--testset 'ocelot'  \
--test-method 'mv' 


python test_rand.py \
--classification-task 'tumor' \
--model 'UNI' \
--crop-size 96 \
--testset 'ocelot' 

##########################################
python train_rand.py \
--classification-task 'tumor' \
--model 'UNI' \
--crop-size 96 \
--testset 'pannuke'

python test_rand.py \
--classification-task 'tumor' \
--model 'UNI' \
--crop-size 96 \
--testset 'pannuke' \
--test-method 'mv' 

python test_rand.py \
--classification-task 'tumor' \
--model 'UNI' \
--crop-size 96 \
--testset 'pannuke' 

# #####################################

python train_rand.py \
--classification-task 'tumor' \
--model 'UNI' \
--crop-size 96 \
--testset 'nucls' \
--multitask '' 

python test_rand.py \
--classification-task 'tumor' \
--model 'UNI' \
--crop-size 96 \
--testset 'nucls'  \
--multitask '' 


# ######################################

python train_rand.py \
--classification-task 'tumor' \
--model 'UNI' \
--crop-size 96 \
--testset 'nucls' 

python test_rand.py \
--classification-task 'tumor' \
--model 'UNI' \
--crop-size 96 \
--testset 'nucls'  \
--test-method 'mv' 

python test_rand.py \
--classification-task 'tumor' \
--model 'UNI' \
--crop-size 96 \
--testset 'nucls' 

## train_rand for single headed tasks
python train_rand.py \
--classification-task 'tumor' \
--model 'UNI' \
--crop-size 96 \
--testset 'pannuke' \
--multitask '' 

python test_rand.py \
--classification-task 'tumor' \
--model 'UNI' \
--crop-size 96 \
--testset 'pannuke' \
--multitask ''  

######################################

python train_rand.py \
--classification-task 'tumor' \
--model 'UNI' \
--crop-size 96 \
--testset 'ocelot' \
--multitask '' 


python test_rand.py \
--classification-task 'tumor' \
--model 'UNI' \
--crop-size 96 \
--testset 'ocelot'  \
--multitask '' 


