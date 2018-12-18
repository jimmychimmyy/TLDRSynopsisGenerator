# “TLDR” Plot Synopsis Generator

### Abstract
The goal of this project was to learn a model that could generate semantically accurate plot synopses from movie plot summaries. Four seq2seq models were trained and their performance was evaluating by comparing the similarity between words in the target sentences and generated sentences. While none of the four models have achieved immediate accurate results, the seq2seq models show great promise for generating semantically accurate plot synopses.

The code used in this project is based on the source code from the Pytorch Sequence to Sequence Translation Tutorial. The code was extended to use a bidirectional encoder.

### Getting Started
```
pip install requirements.txt
python3 main.py
```

### Diagram
![Diagram](https://github.com/jimmychimmyy/tldr_synopsis_generator/diagram.png)
