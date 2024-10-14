python train_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'lizard' \
--crop-size 32

python train_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'cptacCoad' \
--crop-size 32

python train_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'tcgaBrca' \
--crop-size 32

python train_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'nucls' \
--crop-size 32


# train_randstainna for single headed tasks
python train_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'lizard' \
--multitask '' \
--crop-size 32

python train_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'cptacCoad' \
--multitask '' \
--crop-size 32

python train_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'tcgaBrca' \
--multitask '' \
--crop-size 32

python train_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'nucls' \
--multitask '' \
--crop-size 32


# test_randstainna using multitask models with majority vote
python test_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'lizard' \
--test-method 'mv' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'cptacCoad'  \
--test-method 'mv' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'tcgaBrca'  \
--test-method 'mv' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'nucls'  \
--test-method 'mv' \
--crop-size 32


#test_randstainna using multitask model based on clusters
python test_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'lizard' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'cptacCoad' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'tcgaBrca' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'nucls' \
--crop-size 32

# test_randstainna using single headed model
python test_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'lizard' \
--multitask ''  \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'cptacCoad'  \
--multitask '' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'tcgaBrca'  \
--multitask '' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--model 'ResNet50' \
--testset 'nucls'  \
--multitask ''  \
--crop-size 32

