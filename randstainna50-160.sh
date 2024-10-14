
python train_randstainna2.py \
--classification-task 'tumor' \
--testset 'pannuke' \
--model 'ResNet50'

python train_randstainna2.py \
--classification-task 'tumor' \
--testset 'ocelot' \
--model 'ResNet50'

python train_randstainna2.py \
--classification-task 'tumor' \
--testset 'nucls' \
--model 'ResNet50'


# train_randstainna for single headed tasks

python train_randstainna2.py \
--classification-task 'tumor' \
--testset 'pannuke' \
--multitask '' \
--model 'ResNet50'

python train_randstainna2.py \
--classification-task 'tumor' \
--testset 'ocelot' \
--multitask '' \
--model 'ResNet50'

python train_randstainna2.py \
--classification-task 'tumor' \
--testset 'nucls' \
--multitask '' \
--model 'ResNet50'



# test_randstainna using multitask models with majority vote

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'pannuke'  \
--test-method 'mv' \
--model 'ResNet50'

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'ocelot' \
--test-method 'mv' \
--model 'ResNet50'

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'nucls'  \
--test-method 'mv' \
--model 'ResNet50'


#test_randstainna using multitask model based on clusters

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'pannuke' \
--model 'ResNet50'

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'ocelot' \
--model 'ResNet50'

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'nucls' \
--model 'ResNet50'


# test_randstainna using single headed model
python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'pannuke'  \
--multitask ''  \
--model 'ResNet50'

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'ocelot' \
--multitask '' \
--model 'ResNet50'

python test_randstainna2.py \
--classification-task 'tumor' \
--testset 'nucls'  \
--multitask '' \
--model 'ResNet50'