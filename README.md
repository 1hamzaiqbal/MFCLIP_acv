## Overview

This repository contains the implementation of the proposed MF-CLIP and the baseline methods. The implementation is based on a popular open-source tool, CoOp, which can be found at [CoOp GitHub Repository](https://github.com/KaiyangZhou/CoOp). Follow the steps below to reproduce the results of MF-CLIP.

## Instructions

Please refer to `DATASET.MD` for dataset preparation instructions. After setting up the dataset, run the following scripts in order to reproduce MF-CLIP:

1. **Finetune CLIP with Margin-based Loss**  
   ```bash
   ./scripts/finetune.sh
   ```

2. **Train Generator**  
   ```bash
   ./scripts/unet.sh
   ```

3. **Train Target Models**  
   ```bash
   ./scripts/train.sh
   ```

4. **Evaluate Results**  
   ```bash
   ./scripts/eval.sh
   ```

5. **Optional: Obtain Baseline Method Results**  
   ```bash
   ./scripts/baseline.sh
   ```
