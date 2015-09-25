# PLDA Quick Start #
## Introduction ##

PLDA must be run in Linux enviroment with C++ compiler and MPI library installed.

## Installation ##

Download the package `plda+plus.tar.gz`

	tar xzvf plda+plus.tar.gz
	cd plda+plus
	make

There are 3 versions of LDA included in the package. 4 binary files will be generated in the directory: 
`lda`, `mpi_lda`, `mpi_ldaplus` and `infer`.

## A simple training and testing ##

There is a file `test_data.txt` in the package under `plda+plus/testdata` directory.

**Training:**

	./lda --num_topics 2 --alpha 0.1 --beta 0.01 --training_data_file testdata/test_data.txt --model_file /tmp/lda_model.txt --burn_in_iterations 100 --total_iterations 150

or

	mpirun -n 4 ./mpi_lda --num_topics 2 --alpha 0.1 --beta 0.01 --training_data_file testdata/test_data.txt --model_file /tmp/lda_model.txt --total_iterations 150

or

	mpirun -n 4 ./mpi_ldaplus --num_pw--num_topics 2 --alpha 0.1 --beta 0.01 --training_data_file testdata/test_data.txt --model_file /tmp/lda_model.txt --total_iterations 150

for different versions of LDA.

After training completes, a file `/tmp/lda_model.txt` is generated which stores the training result. Each line is the topic distribution of a word. The first element is the word string, the its occurrence count within each topic. A Python script `view_model.py` is used to convert the model to a readable text.

For `mpi_ldaplus`, it will generate `num_pw` files (here named as `lda_model_XXX`, where `XXX = 0,1,...,num_pw-1`). To collect all of the model files and concatenate them into a whole model file, run

	cd /tmp
	cat lda_model.txt_0 lda_model.txt_1 ... lda_model.txt 

**Infer unseen documents:**

	./infer --alpha 0.1 --beta 0.01 --inference_data_file testdata/test_data.txt --inference_result_file /tmp/inference_result.txt --model_file /tmp/lda_model.txt --total_iterations 15 --burn_in_iterations 10

## Command-line flags ##
**Training flags:**

- `alpha`: Suggested to be `50/number_of_topics`
- `beta`: Suggested to be `0.01`
- `num_pw`: The number of pw processors, which should be greater than 0 and less than the total number of processors (here is 6). Suggested to be `number_of_processors/3`. ** This only takes effect for mpi_ldaplus version **.
- `num_topics`: The total number of topics.
- `total_iterations`: The total number of GibbsSampling iterations.
- `burn_in_iterations`: After `--burn_in_iterations` iteration, the model will be almost converged. Then we will average models of the last `(total_iterations-burn_in_iterations)` iterations as the final model. **This only takes effect for single processor version**. For example: you set `total_iterations` to 200, you found that after 170 iterations, the model is almost converged. Then you could set `burn_in_iterations` to 170 so that the final model will be the average of the last 30 iterations.
- `model_file`: The output file of the trained model.
- `training_data_file`: The training data.

## Inferring flags ##

- `alpha` and `beta` should be the same with training.
- `Total_iterations`: The total number of GibbsSampling iterations for an unseen document to determine its word topics. This number needs not be as much as training, usually tens of iterations is enough.
- `burn_in_iterations`: For an unseen document, we will average the `document_topic_distribution` of the last `(total_iterations-burn_in_iterations)` iterations as the final `document_topic_distribution`. 