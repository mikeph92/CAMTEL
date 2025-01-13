python train_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'pannuke'

python test_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'pannuke' \
--test-method 'mv' 

python test_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'pannuke' 

#######################################

python train_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'ocelot' 

python test_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'ocelot'  \
--test-method 'mv' 


python test_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'ocelot' 


#######################################

python train_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'nucls' 

python test_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'nucls'  \
--test-method 'mv' 

python test_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'nucls' 

# train_randstainna for single headed tasks
python train_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'pannuke' \
--multitask '' 
python test_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'pannuke' \
--multitask ''  

# ######################################

python train_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'ocelot' \
--multitask '' 


python test_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'ocelot'  \
--multitask '' 

# #####################################

python train_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'nucls' \
--multitask '' 

python test_no_aug.py \
--classification-task 'tumor' \
--model 'EfficientNet' \
--crop-size 96 \
--testset 'nucls'  \
--multitask '' 

