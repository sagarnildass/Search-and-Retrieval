# Kaggle Notebook Setup for Stage 1 Training

## Quick Start

1. **Create a new Kaggle Notebook**

2. **Add your dataset**:
   - Go to "Add Input" → "Add Dataset"
   - Search for your dataset: `search-and-retrieval-ttsn-training-data`
   - Or upload your data files

3. **Enable GPU**:
   - Go to Settings → Accelerator → GPU T4 x2 (or better)

4. **Install dependencies** (first cell):
```python
!pip install sentence-transformers -q
```

5. **Copy the training script** (second cell):
   - Copy entire contents of `train_stage1_kaggle.py`
   - Paste into a code cell

6. **Run the script**:
   - The script will automatically:
     - Load data from `/kaggle/input/search-and-retrieval-ttsn-training-data/`
     - Train Stage 1 (epochs 1-2)
     - Save checkpoints to `/kaggle/working/checkpoints/`

## Data Structure Expected

Your dataset should have these files:
```
/kaggle/input/search-and-retrieval-ttsn-training-data/
├── training_pairs.parquet
├── val_pairs.parquet
├── test_pairs.parquet
└── training_stats.pkl
```

## Output

Checkpoints will be saved to:
```
/kaggle/working/checkpoints/
├── epoch_1.pt
├── epoch_2.pt
└── best_model.pt
```

## Download Checkpoints

After training completes:
1. Go to "Output" tab
2. Download the `checkpoints/` folder
3. Use `epoch_2.pt` for hard negative mining

## Configuration

You can modify these variables in the script:
- `BATCH_SIZE = 64` - Adjust based on GPU memory
- `LEARNING_RATE = 2e-5` - Learning rate
- `NUM_WORKERS = 2` - Data loader workers

## Tips

- **GPU Memory**: If you get OOM errors, reduce `BATCH_SIZE` to 32 or 16
- **Training Time**: Stage 1 takes ~2-4 hours on GPU T4 x2
- **Monitoring**: Watch the progress bars and loss values
- **Save Early**: Kaggle sessions can timeout, so checkpoints save after each epoch

## Next Steps After Stage 1

1. Download `epoch_2.pt` checkpoint
2. Use it to mine hard negatives (run `hard_negative_mining.py`)
3. Train Stage 2 with new hard negatives
4. Train Stage 3 for fine-tuning
