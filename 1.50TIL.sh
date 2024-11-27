# python train_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'lizard'

python test_randstainna.py \
--classification-task 'TIL' \
--model "ResNet50" \
--testset 'lizard' \
--test-method 'mv'

# python test_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'lizard' 

#########################################
# python train_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'cptacCoad' 

python test_randstainna.py \
--classification-task 'TIL' \
--model "ResNet50" \
--testset 'cptacCoad'  \
--test-method 'mv' 

# python test_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'cptacCoad' 

# #########################################
# python train_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'tcgaBrca' 


# python test_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'tcgaBrca'  \
# --test-method 'mv' 


# python test_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'tcgaBrca' 

# ######################################
# python train_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'nucls' 

# python test_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'nucls'  \
# --test-method 'mv' 

# python test_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'nucls' 

# train_randstainna for single headed tasks
# python train_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'lizard' \
# --multitask '' 

python test_randstainna.py \
--classification-task 'TIL' \
--model "ResNet50" \
--testset 'lizard' \
--multitask ''  

#########################################
# python train_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'cptacCoad' \
# --multitask '' 

python test_randstainna.py \
--classification-task 'TIL' \
--model "ResNet50" \
--testset 'cptacCoad'  \
--multitask '' 

# #####################################
# python train_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'tcgaBrca' \
# --multitask '' 

# python test_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'tcgaBrca'  \
# --multitask '' 


# #########################################
# python train_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'nucls' \
# --multitask ''

# python test_randstainna.py \
# --classification-task 'TIL' \
# --model "ResNet50" \
# --testset 'nucls'  \
# --multitask ''  

