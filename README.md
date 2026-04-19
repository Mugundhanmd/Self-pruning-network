# Self-Pruning Neural Network – CIFAR-10

Name       : Mugundhan R
College    : SRM Institute of Science and Technology
Department : B.Tech CSE (AI & ML) – 3rd Year
Submitted  : Tredence Analytics – AI Intern Case Study (April 2026)

---

## What is This Project?

Normal neural networks are very large and slow.
To make them smaller and faster, we "prune" (remove) 
unnecessary weights.

Usually pruning is done AFTER training.
But in this project, the network learns to prune ITSELF 
during training — automatically.

How?
- Every weight has a "gate" value between 0 and 1
- Gate = 0 means the weight is removed (pruned)
- Gate = 1 means the weight is kept (active)
- The network learns on its own which gates to set to 0

---

## How It Works (Simple)

Step 1: Each weight is multiplied by a gate value
Step 2: If gate is near 0, that weight does nothing
Step 3: We add a penalty (L1 loss) to push gates toward 0
Step 4: The more we increase lambda, the more weights get pruned

Formula:
Total Loss = Classification Loss + Lambda x Sum of all Gates

---

## Files in This Repo

self_pruning_network.py  → Complete Python code
report.md                → Full explanation and analysis
gate_distributions.png   → Shows how many gates went to 0
tradeoff_curve.png       → Shows accuracy vs sparsity graph
README.md                → This file

---

## How to Run

Step 1: Install libraries
pip install torch torchvision matplotlib numpy

Step 2: Run the script
python self_pruning_network.py

Step 3: Output you will see
- Accuracy for each lambda value
- Sparsity percentage (how many weights got pruned)
- Two plots saved as PNG files

---

## Results

| Lambda | Accuracy (%) | Sparsity (%) | Observation          |
|--------|-------------|--------------|----------------------|
| 1e-5   |     __      |     __       | Low pruning          |
| 1e-4   |     __      |     __       | Balanced (best)      |
| 1e-3   |     __      |     __       | High pruning         |

Fill in the actual numbers after running on Google Colab.

---

## Key Takeaway

Low Lambda  = Less pruning,  Higher accuracy
High Lambda = More pruning,  Lower accuracy
Medium Lambda = Best balance between size and accuracy

---

## Tech Stack

- Python 3
- PyTorch
- Torchvision (CIFAR-10 dataset)
- Matplotlib

---

Submitted for Tredence AI Engineering Internship – April 2026
