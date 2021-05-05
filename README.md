# Translatish - Video Translation System


<!--[![Issues](https://img.shields.io/github/issues/Translatish/Translatish)](https://github.com/Translatish/Translatish/)-->
[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
![Maintenance](https://img.shields.io/maintenance/yes/2021)

# **Translatish - Video Translation System**

**A Project Report**

_ **Submitted By:** _

Aanshi Patwari(AU1841004)
Dipika Pawar(AU1841052)
Mayankkumar Tank(AU1841057)
Rahul Chocha(AU1841076)
Yash Patel(AU1841125)

at

![](RackMultipart20210505-4-h5s5ok_html_3d5ba760a68469ba.jpg)

**School of Computer Studies(SCS)**

**Ahmedabad, Gujarat**

# **Table of Contents**

**[1. Introduction](#_n19ofk6b7hy3) 4**

[1.1. Problem Statement](#_9hmt4r8ict4t) 4

[1.2. Project Overview/Specifications](#_c2t9q7xzr2c5) 4

[1.3. Motivation](#_txefla3mh4v6) 4

**[2. Literature Review](#_v931q490r4h8) 5**

[2.1. Existing System](#_qa2ig2k40mji) 5

[2.2. Proposed System](#_qa2ig2k40mji) 6

[2.3. Feasibility Study](#_xelzoxb3ozba) 7

**[3. System Analysis &amp; Design](#_v931q490r4h8) 9**

[3.1. Aim](#_731ybo6xmxso) 9

[3.2. LSTM Architecture details](#_731ybo6xmxso) 9

[3.2.1. Encoder](#_hsjbevaxqifz) 9

[3.2.2. Decoder](#_z91m5nvek35t) 10

[3.2.2.1. Decoder Train Mode](#_k8b6hgstq0fc) 10

[3.2.2.2. Decoder Inference Mode](#_4744w9ac9mq3) 11

[3.3. Impact of system](#_731ybo6xmxso) 12

[3.4. Dataset](#_vsrbvn19cixl) 12

[3.5. Preprocessing](#_nqm1odqvd3bp) 12

[3.6. Word Embedding](#_vsrbvn19cixl) 13

[3.7. Code Snippets(Algorithm and Pseudocode)](#_5434qutzfrpt) 13

[3.8. Testing](#_qy1htl4vfais) 17

[3.9. System Deployment](#_czq1fqg7dudz) 18

**[4. Results](#_hmwwiy73ho1m) 19**

[4.1. Outputs](#_ksu956fn7udj) 19

[4.2. Working end Product (Web app)](#_7nokncch4kac) 21

[4.2.1. Technology Used](#_2webg31va27h) 21

[4.2.1.1. Frontend](#_c2zpc7sqaw60) 21

[4.2.1.2. Backend](#_mljd0jlxxwji) 21

[4.2.2. SnapShots of Web App](#_3vyc3dxilnfk) 21

**[5. Conclusion and Future Work](#_fc9mlx31opeo) 24**

[5.1. Conclusion](#_h7sufavks2a3) 24

[5.2. Future Work](#_h7sufavks2a3) 24

**[6. References](#_v931q490r4h8) 25**

[6.1. Literature Review](#_3bzcv7aewag9) 25

[6.2. LSTM Architecture](#_kqncg8kkr0yi) 25

**[7. Appendix](#_1fece7t2kpn8) 27**

[7.1. List of Images](#_f6s568458407) 27

[7.2. List of Tables](#_fip80w6hvq91) 27

#

# **1. Introduction**

## **1.1. Problem Statement**

- Conversion of videos in English language into our national language( Hindi language) such that it can reach a wider audience of the country.

## **1.2. Project Overview/Specifications**

- The project proposes the use of neural machine translation techniques to convert the videos in English language into Hindi language.
- The technique involves the steps as follows:
  - Input : Video in English language
  - Process :
    - Fetch audio from video
    - Convert audio to text
    - Translate english text to hindi text
  - Output: Hindi text

## **1.3. Motivation**

- India is a multilingual country where more than 1600 languages are spoken belonging to more than 4 different language families; with Hindi being the national language.
- This beauty of language transcends the boundaries of the different states and cultures in our country. Learning a language other than the mother language is a huge advantage. But a path to multilingualism is a never ending process for most of the people of our country.
- Majority of the population of our country finds it difficult to communicate in English and understand English language videos. So, translation of such videos can become useful.
- Furthermore, by converting visual content(i.e. videos) into understandable language(here, hindi) becomes an important step for understanding the relation between visual and linguistic information which are the richest interaction modalities available to humans.

# **2. Literature Review**

## **2.1. Existing System**

- Rule Based Machine Translation(RBMT)
  - These models are based on the idea of use of available dictionaries and rules of a particular language for translating into another language.
  - These models are prepared manually leading to need of large human task force who can understand both the languages( i.e. linguists)
- Example Based Machine Translation(EBMT)
  - These models are based on the idea of using the available set of translations to continue the translation from one language to another.
  - These models have the limitations of having a limited number of examples being available.
- Static Machine Translation(SMT)
  - Word based
    - Comparing similar words in both languages and understanding the pattern by doing the same operation a number of times using the concept of bag-of-words.
    - This model had a limitation of not being able to consider the order of the words within a sentence,
  - Phrase based
    - Comparing similar phrases in both languages and understanding the pattern by doing the same operation a number of times using the n-gram model.
    - This model also had limitations of not being able to consider the order of phrases within a sentence.
  - Syntax based
    - Use of the subject, the predicate, other parts of the sentence, and then to build a sentence tree using language syntax along with rest of word and phrase translation for translation from one language to another.
    - Complexity of model increased with large amount of dataset consisting of numerous sentences.
- Neural Machine Translation(NMT)
  - Recurrent Neural Networks(RNN)
    - RNN are used to reorder the source to target language translation by the help of semi-supervised learning methods.
    - Word embedding is used to generate the vector value of the words which can be used for translation purposes.
    - RNN are based on the idea of binary tree structure for mapping the word in the target language with the vector values of the source language.
    - This model has a complex structure which in turn leads to more number of computations.
  - Long Short-Term Memory Networks(LSTM)
    - LSTM works on the idea of sequence by sequence learning.
    - It uses a NMT model which has eight layers of encoders and decoders within a deep neural network.
    - It is also known as Bidirectional RNN because of the use of encoder and decoder structure.
    - Encoder-Decoder structures are jointly trained to maximize the conditional probabilities of the target sequence given the input sequence.

## **2.2. Proposed System**

- LSTM model
  - This model is a special kind of RNN capable of learning long-term dependencies.
  - The model mainly consists of steps:
    - Decide the information to pass through the cell state
    - Decide what information to store in the cell state
    - Update the cell state by adding new information and discarding the information no longer needed
    - Predict the output based on the information

_Figure 1-LSTM architecture._ ![](RackMultipart20210505-4-h5s5ok_html_197acd18a683d56f.png)

## **2.3. Feasibility Study**

- Comparison between RNN and LSTM
  - Gated Recurrent Units(GRU) gating values are close to LSTM values with lesser run time in comparison to RNN.
  - The LSTM layer architecture is built in a way such that the network &quot;decides&quot; whether to modify its &quot;internal memory&quot; or not, at each step.
  - Doing so, and if properly trained, then the layer can keep track of important events from further in the past, allowing for much richer inference.
  - LSTM networks were made with the purpose to solve long term dependency problems that traditional RNNs have.
  - LSTM networks are specially made to solve long-term dependency problems in RNNs and they are super good at making use of data from a while ago (inherent in its nature) by using cell states and RNN has a complex structure which in turn has more computations therefore making LSTM more preferred over RNN.

# **3. System Analysis &amp; Design**

## **3.1. Aim**

To implement a model which can perform word-by-word translation of a given english sentence into hindi with accurate results.

## **3.2. LSTM Architecture details**

- The LSTM reads data one sequence after the other.
- If the input is of length &#39;k&#39; then it will read it ask time-stamps
- The main aim is to bring long and short term dependencies
- The whole system is simplified by 3 Gates and 1 state:
  - Forget Gate: This gate will decide that information should be passed from previous state (h(t-1)) to next state (h(t)) or not
  - Input Gate: This gate deals with the data to update the cell states
  - Output Gate: This decides the information that is relevant to next hidden state
  - Cell state: The pointwise addition of the output of Forget and Input gate updates cell state
- Input tokens are represented by {x1,x2,...,xk}, hidden states by {h1,h2,...,hk}, cell states by {c1,c2,...,c3} and output token by {y1,y2…,yk}.

LSTM consists of 3 neural network architectures.

- Encoder
- Decoder Train Mode
- Decoder Inference Mode

###

### **3.2.1. Encoder**

- Encoder basically summaries the whole sentence
- The inputs for the encoder are feature vectors of each word of the sentence which is shown in the figure
- After embedding inputs to the encoder, it will generates hidden and cell states at each time and gives final outputs y1, y2, ..yk
- In this project, the hidden and cell states are only stored and outputs are discarded because the states are helpful to store the summarized sequence till the previous state. For this application the states are only needed

![](RackMultipart20210505-4-h5s5ok_html_56f1546a837ea60d.png)

_Figure 2-Encoder Model._

### **3.2.2. Decoder**

#### 3.2.2.1. Decoder Train Mode

- This accepts the final states of encoder as the input states
- Then getting word start\_ and it starts generate word
- At the next instance, We give actual Hindi translation words as input form training labels and it will again predict the next word (Shown in the below figure).
- This uses &quot;Teacher forcing method&quot;

![](RackMultipart20210505-4-h5s5ok_html_c1f06141e9d0941a.jpg)

_Figure 3-Decoder Train M __odel__._

#### 3.2.2.2. Decoder Inference Mode

- In this application we need slightly different architecture for output testing Decoder.
- As we don&#39;t have labels for testing purposes, we need to pass the predicted word to the next timestamp&#39;s input (Shown in the below figure).

![](RackMultipart20210505-4-h5s5ok_html_2a1ad904b219ccc8.jpg)

_Figure 4- __Decoder Inference Model__._

## **3.3. Impact of system**

- Video marketing can be even more effective overseas.
- Translation helps in cutting across language barriers and interacting with people in such countries.

## **3.4. Dataset**

- Bilingual corpora
- Part of parallel corpora which contains multiple corporas.
- Taken from about 37726 TED talks, news articles, Wikipedia articles.
- Has unique 1,24,318 english records and 9,7662 hindi records
- Corpus link: [HindiEnglishCorpora](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0023-625F-0)
- Cleaned dataset
- It is sentence tokenized (sentence aligned)

![](RackMultipart20210505-4-h5s5ok_html_892918f045fe66a4.png)

_Figure 5-Dataset Sample._

## **3.5. Preprocessing**

- Removal of duplicate and null records
- Conversion to lowercase
- Removal of Quotes, Special characters, extra spaces
- Computing the length of english and corresponding Hindi sentence, append it in the dataset
- Dictionary of all the words of Hindi and English
- Adding &#39;START\_&#39; and &#39;\_END&#39; tokens to the Hindi labels, such that decoder can understand starting and end of the sentences

## **3.6. Word Embedding**

- For feature vector representation, there are many techniques like word embedding, word2vect, etc.
- Here word embedding is used to represent word features and it can be implemented using Keras API in python
- In embedding, there are certain features which are extracted automatically from the text. In which if the number of words is n and the number of features is m then the matrix will be the size of mxn

![](RackMultipart20210505-4-h5s5ok_html_aa8ab437f8e9b311.png)

_Figure 6-Word Embedding Sample._

- In the above example, n=10K and m=50.
- The features are like gender, royal, kind etc.
- For word which having any particular feature has High value in the matrix
- As King, Queen, Man, Woman all can relate to Gender. So their corresponding values are High but Man is not related to Royal. So that value is near to zero

## **3.7. Code Snippets(Algorithm and Pseudocode)**

- Pre-processing

![](RackMultipart20210505-4-h5s5ok_html_487a875e9786d08f.png)

_Figure 7 __-Preprocessing Code Snippet__._

- Batch Generation:

![](RackMultipart20210505-4-h5s5ok_html_c3bf5a8cf9f8a2b.png)

_Figure 8-Batch Generation Pseudocode._

![](RackMultipart20210505-4-h5s5ok_html_68a907dadf88f8fd.png)

_Figure 9-Batch G __eneration Code Snippet__._

- Model Building

![](RackMultipart20210505-4-h5s5ok_html_b1ae1bdbf922d9d3.png)

_Figure 10-M __odel Building Code Snippet__._

- Model Summary

![](RackMultipart20210505-4-h5s5ok_html_ab8b7e70174c8c9d.png)

_Figure 11-Model Summary._

- BLEU score computation
  - For n-gram precision calculation:

![](RackMultipart20210505-4-h5s5ok_html_ffaef59b947c2df.png)

_Figure 12-n gram precision pseudocode._

  - Cumulative n-gram score:

![](RackMultipart20210505-4-h5s5ok_html_ff60a83cfb71754c.png)

_Figure_ _13-Cumulative n-gram score pseudocode__._

![](RackMultipart20210505-4-h5s5ok_html_8143dbd1fdfdd56e.png)

_Figure 14-BLEUscore code snippet._

## **3.8. Testing**

For testing the applications like Neural Machine Translation, BLEU algorithm is used

BLEU score:

- Bilingual evaluation understudy
- Number between 0-1
- The number indicates how much similar the predicted text is to reference text
- 1 indicates the maximum similarity
- It can be using Unigram, bigram,... or the combination of any(Here 2 and 4-gram cumulatives are used)
- The algorithm is mentioned above

Example: **Reference 1** : The cat is on the mat.

**Reference 2** : There is a cat on the mat.

**Predicted Output** : The cat the cat on the mat. (Target)

Bigram BLEU Score Computation

| **Possible Bigrams** | **Frequency in**  **Target Sentence** | **Actual frequency from the reference sentences** |
| --- | --- | --- |
| the cat | 2 | 1 |
| cat the | 1 | 0 |
| cat on | 1 | 1 |
| on the | 1 | 1 |
| the mat | 1 | 1 |

_Table 1-BL __EU score computation example table__._

- Our total count is 2+1+1+1+1=6
- Actual count is 1+0+1+1+1=4
- Bleu score = count/total = 4/6

## **3.9. System Deployment**

As the system is divided into steps:

- Uploading the video in English language
- Video to audio conversion
- Audio to text conversion
- English text to Hindi translation
- Displaying the Translation

# **4. Results**

## **4.1. Outputs**

- Preprocessing Results

![](RackMultipart20210505-4-h5s5ok_html_a793ea2e55b1c04d.png)

_Figure 15-Preprocessing output._

- Model Training
  - After training the model for 50 epochs, it starts to converge which suggests that the training should be done till 50 epochs.

![](RackMultipart20210505-4-h5s5ok_html_b763c55ad31fe4d7.jpg)

_Figure 16-Trained M __odel Output__._

- Model Validation
  - After 30 epochs the model stops learning which suggests to stop the training of the model at 30 epochs.

![](RackMultipart20210505-4-h5s5ok_html_2297984387c720f6.jpg)

_Figure 17-Validation Curve output._

- Model Testing

![](RackMultipart20210505-4-h5s5ok_html_5eaa26301e3f2dad.jpg)

_Figure 18-Te__st_ _Example1._

Here, the model is able to predict the hindi translation for the given english sentence exactly similar to the actual hindi translation.

![](RackMultipart20210505-4-h5s5ok_html_bcba3cdeb7adc8e1.jpg)

_Figure 19-Test Example2._

Here, the model is able to predict the hindi translation for the given hindi translation with a bit difference with the actual hindi translation due to difference in semantics and syntactics of the sentence

- Bleu Score

| Sr. no. | Dataset Name | Description | BLEU Score |
| --- | --- | --- | --- |
| 1. | [HindiEnglishCorpora](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0023-625F-0) | Contains TEDtalks, news articles and Wikipedia articles | 53.01% |
| 2. | [Machine-Translation-English-To-Hindi](https://github.com/suvaansh/Machine-Translation-English-to-Hindi-/blob/master/hin.txt) | Contains day-to-day routinely used sentences | 36.9% |
| 3. | [IITBombay English-Hindi Corpus Dataset](https://metatext.io/datasets/iit-bombay-english-hindi-corpus) | Contains Indian judicial statements and their corresponding translated sentences | 12% |

_Table 2-Test Dataset Table._

## **4.2. Working end Product (Web app)**

### **4.2.1. Technology Used**

#### 4.2.1.1. Frontend

- HTML
- CSS
- React JS

#### 4.2.1.2. Backend

- Django
- Django Rest-frameworks
- Python
- Tensorflow
- Keras
- Audiopy, pydub

### **4.2.2. SnapShots of Web App**

- Welcome (Intro) Screen **:**

This is the intro screen when you reach out to our web app.

![](RackMultipart20210505-4-h5s5ok_html_4bdc26615165ac2.png)

_Figure 20-Front Screen._

- About Us **:**

About us, the screen includes the details of the project member with their profile pictures. (Meet the Team)

![](RackMultipart20210505-4-h5s5ok_html_5061d924ad10d974.png)

_Figure 21-About us_ _Screen__._

- Explore Page **:**

This is the main screen where you can do your desired work (video translation). First snapshot showing the page without uploading the video. There you can see that there is an option of uploading the video with click or drag and drop facility. Talking about the second snapshot where video is uploaded and there you can play the video. And at the end of the page you can see two divisions one shows the extracted english sentences (converted from the video). And Another one shows the translated hindi text of the video.

![](RackMultipart20210505-4-h5s5ok_html_252e6a354585bafc.png)

_Figure 22-Explore Screen output 1._

![](RackMultipart20210505-4-h5s5ok_html_474fdf298b0ef14d.png)

_Figure 23-Explore Screen output 2._

# **5. Conclusion and Future Work**

## **5.1. Conclusion**

- The implementation of LSTM model for the purpose of video translation i.e. neural machine translation gave accurate results for videos similar to the training dataset
- There was a bit of fluctuation in the results for translation of videos different from the training dataset type.

## **5.2. Future Work**

- Implementation of proposed model on different datasets
- Try out translation into other languages
- Train the model on dataset for variety of sentences from different videos
- Conversion of the translated sentences into video format
- Improve the model performance by implementing it on super computers which can provide more efficient results and computation power
- Change the speech to text conversion strategy.

# **6. References**

## **6.1. Literature Review**

[1] Translartisan. (2018, August 31). Rule-based machine translation. Retrieved from [https://translartisan.wordpress.com/tag/rule-based-machine-translation/](https://translartisan.wordpress.com/tag/rule-based-machine-translation/)

[2] A., P., &amp; P. (2018). The IIT Bombay English-Hindi Parallel Corpus. 1-4. Retrieved May 19, 2018, from [https://arxiv.org/pdf/1710.02855.pdf](https://arxiv.org/pdf/1710.02855.pdf).

[3] Saini, S., &amp; Sahula, V. (2018). Neural Machine Translation for English to Hindi. _2018 Fourth International Conference on Information Retrieval and Knowledge Management (CAMP)_. doi:10.1109/infrkm.2018.8464781

[4] S. P. Singh, A. Kumar, H. Darbari, L. Singh, A. Rastogi and S. Jain, &quot;Machine translation using deep learning: An overview,&quot; 2017 International Conference on Computer, Communications and Electronics (Comptelix), 2017, pp. 162-167, doi: 10.1109/COMPTELIX.2017.8003957.

## **6.2. LSTM Architecture**

[5] Lamba, H. (2019, February 17). Word Level English to Marathi Neural Machine Translation using Seq2Seq Encoder-Decoder LSTM Model. Retrieved from [https://towardsdatascience.com/word-level-english-to-marathi-neural-machine-translation-using-seq2seq-encoder-decoder-lstm-model-1a913f2dc4a7](https://towardsdatascience.com/word-level-english-to-marathi-neural-machine-translation-using-seq2seq-encoder-decoder-lstm-model-1a913f2dc4a7)

[6] Ranjan, R. (2020, January 16). Neural Machine Translation for Hindi-English: Sequence to sequence learning. Retrieved from [https://medium.com/analytics-vidhya/neural-machine-translation-for-hindi-english-sequence-to-sequence-learning-1298655e334a](https://medium.com/analytics-vidhya/neural-machine-translation-for-hindi-english-sequence-to-sequence-learning-1298655e334a)

[7] V, B. (2020, November 16). A Comprehensive Guide to Neural Machine Translation using Seq2Sequence Modelling using PyTorch. Retrieved from [https://towardsdatascience.com/a-comprehensive-guide-to-neural-machine-translation-using-seq2sequence-modelling-using-pytorch-41c9b84ba350](https://towardsdatascience.com/a-comprehensive-guide-to-neural-machine-translation-using-seq2sequence-modelling-using-pytorch-41c9b84ba350)

[8] Understanding LSTM Networks. (n.d.). Retrieved from [https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[9] Pedamallu, H. (2020, November 30). RNN vs GRU vs LSTM. Retrieved from [https://medium.com/analytics-vidhya/rnn-vs-gru-vs-lstm-863b0b7b1573](https://medium.com/analytics-vidhya/rnn-vs-gru-vs-lstm-863b0b7b1573)

[10] Mittal, A. (2019, October 12). Understanding RNN and LSTM. Retrieved from [https://aditi-mittal.medium.com/understanding-rnn-and-lstm-f7cdf6dfc14e](https://aditi-mittal.medium.com/understanding-rnn-and-lstm-f7cdf6dfc14e)

[11] Culurciello, E. (2019, January 10). The fall of RNN / LSTM. Retrieved from [https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0](https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0)

# **7. Appendix**

## **7.1. List of Images**

|
- [Figure 1-LSTM architecture](#kix.kb0eoujdsjfl)
- [Figure 2-Encoder Model.](#1dk3r8l48ilh)
- [Figure 3-Decoder Train Model.](#2dncut4xbxjy)
- [Figure 4-Decoder Inference Model.](#muqsxnr5o8c3)
- [Figure 5-Dataset Sample.](#u4zhmtua2uzj)
- [Figure 6-Word Embedding Sample.](#7zwtkyejorp)
- [Figure 7-Preprocessing Code Snippet.](#26ee0dg0bxrh)
- [Figure 8-Batch Generation Pseudocode.](#5i6p7du77lxq)
- [Figure 9-Batch Generation Code Snippet.](#xoobqmk5d0j1)
- [Figure 10-Model Building Code Snippet.](#ar5bqlswfrl)
- [Figure 11-Model Summary.](#ug62oh2nzly7)
- [Figure 12-n gram precision pseudocode.](#lsrk84eo6z8b)
- [Figure 13-Cumulative n-gram score pseudocode.](#w2q0eapzbku1)
- [Figure 14-BLEUscore code snippet.](#fye7ydo0tzw4)
- [Figure 15-Preprocessing output.](#dthxx7xrjoh)
- [Figure 16-Trained Model Output.](#gdvkifop8vkm)
- [Figure 17-Validation Curve output.](#9dp35ktxnq9y)
- [Figure 18-Test Example1.](#cv94czt5rr6p)
- [Figure 19-Test Example2.](#51sy2hkit84x)
- [Figure 20-Front Screen.](#l4165lw76qz0)
- [Figure 21-About us Screen.](#6hl7hf6d4qm6)
- [Figure 22-Explore Screen output 1.](#a3w6307v93yb)
- [Figure 23-Explore Screen output 2.](#fhbhk55frpgc)
 |
| --- |

## **7.2. List of Tables**

|
- [Table 1-BLEU score computation example table.](#g9gx84wfpayj)
- [Table 2-Test Dataset Table.](#3dpe9rygpoda)
 |
| --- |

Page - 24 of 27

[![Stargazers repo roster for @Translatish/Translatish](https://reporoster.com/stars/Translatish/Translatish)](https://github.com/Translatish/Translatish/stargazers)

[![Forkers repo roster for @Translatish/Translatish](https://reporoster.com/forks/Translatish/Translatish)](https://github.com/Translatish/Translatish/network/members)



