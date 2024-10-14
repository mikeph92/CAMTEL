
python train_randstainna2.py \
--classification-task 'tumor' \
--testset 'pannuke' 

python train_randstainna2.py \
--classification-task 'tumor' \
--testset 'ocelot' 

python train_randstainna2.py \
--classification-task 'tumor' \
--testset 'nucls' 


# train_randstainna for single headed tasks
python train_randstainna2.py \
--classification-task 'tumor' \
--testset 'pannuke' \
--multitask ''

python train_randstainna2.py \
--classification-task 'tumor' \
--testset 'ocelot' \
--multitask ''

python train_randstainna2.py \
--classification-task 'tumor' \
--testset 'nucls' \
--multitask ''



# test_randstainna using multitask models with majority vote

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'pannuke'  \
--test-method 'mv'

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'ocelot' \
--test-method 'mv'

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'nucls'  \
--test-method 'mv'


#test_randstainna using multitask model based on clusters

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'pannuke' 

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'ocelot'

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'nucls' 


# test_randstainna using single headed model
python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'pannuke'  \
--multitask ''

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'ocelot' \
--multitask ''

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'nucls'  \
--multitask ''
