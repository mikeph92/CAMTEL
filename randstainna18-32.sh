python train_randstainna2.py \
--classification-task 'TIL' \
--testset 'lizard' \
--crop-size 32

python train_randstainna2.py \
--classification-task 'TIL' \
--testset 'cptacCoad' \
--crop-size 32

python train_randstainna2.py \
--classification-task 'TIL' \
--testset 'tcgaBrca' \
--crop-size 32

python train_randstainna2.py \
--classification-task 'TIL' \
--testset 'nucls' \
--crop-size 32


# train_randstainna for single headed tasks
python train_randstainna2.py \
--classification-task 'TIL' \
--testset 'lizard' \
--multitask '' \
--crop-size 32

python train_randstainna2.py \
--classification-task 'TIL' \
--testset 'cptacCoad' \
--multitask '' \
--crop-size 32

python train_randstainna2.py \
--classification-task 'TIL' \
--testset 'tcgaBrca' \
--multitask '' \
--crop-size 32

python train_randstainna2.py \
--classification-task 'TIL' \
--testset 'nucls' \
--multitask '' \
--crop-size 32


# test_randstainna using multitask models with majority vote
python test_randstainna2.py \
--classification-task 'TIL' \
--testset 'lizard' \
--test-method 'mv' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--testset 'cptacCoad'  \
--test-method 'mv' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--testset 'tcgaBrca'  \
--test-method 'mv' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--testset 'nucls'  \
--test-method 'mv' \
--crop-size 32


#test_randstainna using multitask model based on clusters
python test_randstainna2.py \
--classification-task 'TIL' \
--testset 'lizard' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--testset 'cptacCoad' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--testset 'tcgaBrca' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--testset 'nucls' \
--crop-size 32

# test_randstainna using single headed model
python test_randstainna2.py \
--classification-task 'TIL' \
--testset 'lizard' \
--multitask ''  \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--testset 'cptacCoad'  \
--multitask '' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--testset 'tcgaBrca'  \
--multitask '' \
--crop-size 32

python test_randstainna2.py \
--classification-task 'TIL' \
--testset 'nucls'  \
--multitask ''  \
--crop-size 32

