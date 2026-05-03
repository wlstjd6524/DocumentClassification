import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SWIN_LOGITS = "/Users/caroline/Documents/swin_base_384_v1_infer/predict_logits.pt"
CONV_LOGITS = "/Users/caroline/Documents/convnext_base1/predict_logits_convnext_base.pt"
SAMPLE_CSV  = "/Users/caroline/data/sample_submission.csv"

OUT_DIR = "/Users/caroline/Documents/ensemble_outputs_conv_dominant"
os.makedirs(OUT_DIR, exist_ok=True)

swin = torch.load(SWIN_LOGITS, map_location="cpu")
conv = torch.load(CONV_LOGITS, map_location="cpu")
assert swin.shape == conv.shape, (swin.shape, conv.shape)

sub_template = pd.read_csv(SAMPLE_CSV)

# Conv 중심 sweep: 0.75 ~ 1.00 (step 0.05)
for w_conv in np.round(np.arange(0.75, 1.001, 0.05), 2):
    w_swin = 1.0 - float(w_conv)

    final_logits = w_conv * conv + w_swin * swin
    preds = torch.argmax(F.softmax(final_logits, dim=1), dim=1).numpy()

    out = sub_template.copy()
    out["target"] = preds

    out_path = os.path.join(OUT_DIR, f"submission_conv{w_conv:.2f}_swin{w_swin:.2f}.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

print("Done ->", OUT_DIR)