# Seq2Seq
##### The seq2seq models encodes the content of an article (encoder input) and one character (decoder input) from the summarized text to predict the next character in the summarized text

##### The implementation can be found in keras_text_summarization/library/seq2seq.py

 > There are three variants of seq2seq model implemented for the text summarization

 - Seq2SeqSummarizer (one hot encoding)
    * training: run demo/seq2seq_train.py
    * prediction: demo code is available in demo/seq2seq_predict.py
 - Seq2SeqGloVeSummarizer (GloVe encoding for encoder input)
    * training: run demo/seq2seq_glove_train.py
	* prediction: demo code is available in demo/seq2seq_glove_predict.py
 - Seq2SeqGloVeSummarizerV2 (GloVe encoding for both encoder input and decoder input)
    * training: run demo/seq2seq_glove_v2_train.py
    * prediction: demo code is available in demo/seq2seq_glove_v2_predict.py


## Train Deep Learning model
##### To train a deep learning model, say Seq2SeqSummarizer, run the following commands:

pip install requirements.txt

cd demo
python seq2seq_train.py 


##### After the training is completed, the trained models will be saved as cf-v1-. in the video_classifier/demo/models.

## Summarization
#### To use the trained deep learning model to summarize an article, run following commands - 
 
cd demo
python seq2seq_predict.py