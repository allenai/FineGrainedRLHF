# fgrlhf

install requirements

```bash
# create conda environment with python 3.9
conda create --name py39 python=3.9
conda activate py39 

# install packages
git clone https://github.com/allenai/FineGrainedRLHF.git
cd FineGrainedRLHF
pip install -e .
python -m spacy download en_core_web_sm
```

### Notice: should move all the py and yml files in the examples/qa_feedback folder. Now haven't finalized path etc. yet so I haven't done that

## Usage
Customize rewards and evaluation metrics for each task in `reward.py`

Dataset is customized in `train_baseline.py` and `train_finegrained.py`

Specify reward model path in yml files

## Run

```bash
bash train_finegrained.sh
bash train_baseline.sh
```
