Use Case: Neuron Shapley

Steps include:
1. Download InceptionV3 model checkpoint.
2. Collection of imagenet dataset:
These has collected images of 10 classe each class contain about 25 images.
3. Execute the cb_run.py file:
Execution of this file create result files(like Players.txt and .h5) which includes players information like name of players(Filters),
and model perfomance for each players.
4. Execute the cb_aggregate.py file:
Execution of this file create result files(chosen_players.txt,vals.txt,variances.txt and counts.txt)these file contain
information about selected players(Filters) like their shapley values,variation of this values etc.
5. After getting result related to selected filters we have used to identify most responsible or least responsible filters
based on performance in prediction of accuracy by just executing inception_helper.py file.
6. After removing least responsible neurons(without much degradation in accuracy) we have saved the checkpoint.
