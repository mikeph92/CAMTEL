python train_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'pannuke'

python test_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'pannuke' \
--test-method 'mv' 

python test_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'pannuke' 

#######################################

python train_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'ocelot' 

python test_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'ocelot'  \
--test-method 'mv' 


python test_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'ocelot' 


#######################################

python train_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'nucls' 

python test_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'nucls'  \
--test-method 'mv' 

python test_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'nucls' 

# train_randstainna for single headed tasks
python train_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'pannuke' \
--multitask '' 

python test_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'pannuke' \
--multitask ''  

######################################

python train_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'ocelot' \
--multitask '' 


python test_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'ocelot'  \
--multitask '' 

#####################################

python train_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'nucls' \
--multitask '' 

python test_randstainna.py \
--classification-task 'tumor' \
--model 'ResNet18' \
--crop-size 96 \
--testset 'nucls'  \
--multitask '' 

