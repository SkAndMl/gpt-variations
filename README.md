## gpt-variations

### Replication
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python fineweb.py
torchrun --standalone --nproc_per_node=<n_gpus> train.py
python evals.py
```