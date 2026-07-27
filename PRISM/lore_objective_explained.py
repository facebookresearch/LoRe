"""
A BAREBONES, HEAVILY-COMMENTED walkthrough of the LoRe learning objective.

This is a teaching file. It is NOT used by the real pipeline. It rebuilds the
important 30 lines of `utils.py` (class `LoRe_regularized`) in miniature so you can
read it top to bottom, run it, and watch the shapes. Once it clicks here, open
`utils.py` lines 189-291 and you'll recognise every line.

You can run it:   uv run python PRISM/lore_objective_explained.py
It uses tiny fake numbers so nothing needs a GPU or the real data.

--------------------------------------------------------------------------------
THE THEORY FIRST (in your own words, made precise)
--------------------------------------------------------------------------------
- Skywork (a reward model) is a Llama. Same transformer body as an LLM; the only
  difference is the head. We don't even use its head here. We just take the last
  token's last-layer hidden state: a vector of 4096 numbers. Call it an EMBEDDING.
  It is the model's summary of a whole (prompt + response) conversation.

- LoRe's claim: a person's reward for a response = that embedding, projected onto a
  small set of K "taste directions" (the BASIS), then mixed together using that
  person's personal recipe (their WEIGHTS).

    reward = embedding  ->  K basis scores  ->  weighted sum  ->  one number

- Two things are LEARNED:
    V : the basis. A matrix of shape [4096, K]. Each of its K columns is one
        4096-long "taste direction". Shared by everyone.
    W : the weights. One row per user, K numbers per row: how much that user
        cares about each of the K directions. Personal to each user.

--------------------------------------------------------------------------------
SHAPE GLOSSARY  (a "shape" is just how many numbers, along each axis)
--------------------------------------------------------------------------------
    F = number of features in an embedding      (real value: 4096)
    K = number of basis directions              (e.g. 1, 5, 10, ...)
    C = number of users   (the code calls it num_classes)
    N = number of training examples, all users pooled together

Below I use tiny values (F=4, K=2, C=3) so every matrix is small enough to read.
"""

import torch
import torch.nn.functional as F_nn   # "nn.functional" holds softmax, logsigmoid, etc.

torch.manual_seed(0)  # make the random numbers repeat, so runs are identical


# =============================================================================
# STEP 1 - WHAT THE DATA LOOKS LIKE
# =============================================================================
# A training example is ONE preference: "for this prompt, the user chose response
# A over response B". We already have both responses as embeddings. The single
# most important trick in this whole objective:
#
#       we store the DIFFERENCE   x = embedding(chosen) - embedding(rejected)
#
# (In the real code this happens in train_basis_full.py line 22.)
#
# Why the difference? Because a reward is linear: reward(r) = embedding(r) . w
# (a dot product with some reward direction w). So:
#
#   reward(chosen) - reward(rejected)
#        = embedding(chosen).w - embedding(rejected).w
#        = ( embedding(chosen) - embedding(rejected) ) . w
#        =  x . w
#
# So the difference vector x already contains everything we need: if x.w is
# POSITIVE, the model scores the chosen response higher, which is what we want.

F, K, C = 4, 2, 3  # tiny: 4 features, 2 basis directions, 3 users

# Fake data: each user has 2 preference examples. Each example is a difference
# vector of length F=4. In reality these are (chosen - rejected) Skywork embeddings.
user_data = [
    torch.randn(2, F),   # user 0: shape [2, 4]  = 2 examples, 4 features each
    torch.randn(2, F),   # user 1
    torch.randn(2, F),   # user 2
]

# The real code "packs" all users into one big stack, and remembers which row
# belongs to which user. (utils.py _prepare_batch, lines 208-221.)
X_cat = torch.cat(user_data, dim=0)          # shape [N, F] = [6, 4]. All examples stacked.
y = torch.cat([                              # shape [N] = [6]. y[n] = which user example n is.
    torch.full((d.shape[0],), i) for i, d in enumerate(user_data)
]).long()
# Now: X_cat[n] is the n-th difference vector, and y[n] says which user owns it.
# e.g. y = [0,0,1,1,2,2]  -> first two rows are user 0's, next two user 1's, etc.


# =============================================================================
# STEP 2 - THE LEARNED PARAMETERS  (V and W)
# =============================================================================
# "Parameter" = a bag of numbers PyTorch will nudge during training to lower the
# loss. We start them at random and let training shape them.

V = torch.randn(F, K, requires_grad=True)    # the basis:  [4, 2]. 2 taste directions.
W = torch.rand(C, K, requires_grad=True)     # the weights:[3, 2]. one row per user.
# requires_grad=True just means "track how the loss depends on these, so we can
# improve them". (In utils.py these are nn.Parameter, same idea: lines 204-205.)

# V_sft = the base reward model's own single direction (its head weight, 4096 long).
# The regulariser will try to keep our basis pointing near it. Fake one here:
V_sft = torch.randn(F, 1)                     # shape [4, 1]


# =============================================================================
# STEP 3 - THE FORWARD PASS: turn (V, W, data) into a score per example
# =============================================================================
# This mirrors utils.py `_forward_from_packed`, lines 223-234. Read the shapes.

# 3a. Make each user's weights a PROBABILITY DISTRIBUTION with softmax.
#     softmax makes the K numbers in each row non-negative and sum to 1.
#     So a user's recipe becomes e.g. [0.7, 0.3]: "70% direction-0, 30% direction-1".
#     (This is the "simplex" constraint. It is ALSO a cause of basis collapse,
#      because it pushes each user toward putting all their weight on ONE column.)
W_row = F_nn.softmax(W, dim=1)                # shape [C, K] = [3, 2]. dim=1 => across the K columns.

# 3b. Build each user's personal reward direction by mixing the basis columns
#     with that user's weights.  V @ W_row.T   is matrix multiplication.
#         V     is [F, K]
#         W_row.T is [K, C]   (.T = transpose = flip rows/columns)
#         result  is [F, C]   -> column c is user c's own 4-long reward direction.
Vw = V @ W_row.T                              # shape [F, C] = [4, 3]

# 3c. Score every example against every user's reward direction.
#         X_cat is [N, F]
#         Vw    is [F, C]
#         result is [N, C]  -> entry [n, c] = "example n scored by user c's reward"
#     The /100 is just numerical scaling (keeps the numbers in a gentle range).
logits_all = (X_cat @ Vw) / 100.0             # shape [N, C] = [6, 3]

# 3d. But example n should only be judged by ITS OWN user's reward (y[n]).
#     `gather` picks, from each row n, the column y[n]. So we keep the diagonal
#     that matches each example to its owner.
logits = logits_all.gather(1, y.unsqueeze(1)).squeeze(1)   # shape [N] = [6]
# logits[n] is now exactly   x_n . (user y[n]'s reward direction) / 100
# Positive => that user's reward ranks the chosen response above the rejected one.


# =============================================================================
# STEP 4 - THE LOSS: two pieces added together
# =============================================================================

# 4a. DATA TERM (utils.py line 234).  This is the Bradley-Terry preference model:
#     P(chosen preferred) = sigmoid(logit).  logsigmoid is log of that.
#     We want that probability HIGH, i.e. logsigmoid HIGH, i.e. -logsigmoid LOW.
#     .mean() averages over all N examples.  <-- note: divides by N, not by K.
nll = -F_nn.logsigmoid(logits).mean()         # a single number. "how wrong are we"

# 4b. ALIGNMENT TERM / the regulariser (utils.py lines 239-242).
#     "Keep each basis column pointing in roughly the same direction as V_sft."
#     cos(a, b) = 1 when two directions are identical, 0 when perpendicular.
V_norm     = F_nn.normalize(V, dim=0)         # make each column length-1
V_sft_norm = F_nn.normalize(V_sft, dim=0)     # make V_sft length-1
cos_sim = (V_norm * V_sft_norm).sum(dim=0)    # shape [K] = [2]. one cosine PER column.
reg = torch.mean(1 - cos_sim)                 # a single number. 0 = perfectly aligned.
# ****** THIS IS RACHEL'S POINT ******
# reg is a MEAN over the K columns -> it secretly divides by K.
# So when we do   alpha * reg   below, each column really feels a pull of  alpha / K.
# The data term (nll) divided by N, NOT by K. So as K grows, the alignment pull per
# column shrinks relative to the data fit. That is the entire "alpha/K" effect.

alpha = 10000.0                               # the real code uses 1e4. A big leash-strength.
total_loss_V = nll + alpha * reg              # utils.py line 289
# Read it as:   fit-the-preferences   +   (alpha/K) * stay-near-the-base-direction


# =============================================================================
# STEP 5 - TRAINING: nudge V and W to make the loss smaller, over and over
# =============================================================================
# The real code (utils.py 272-291) ALTERNATES:
#   - update W while V is held still,
#   - then update V while W is held still.
# and it RAMPS alpha up gradually ("warmup", lines 256-261) so the model first
# learns to fit the data, then slowly feels the alignment leash. Below is the
# bare mechanical loop, minus those refinements, just so you see what "training" is.

opt = torch.optim.Adam([V, W], lr=0.1)        # Adam = the rule for how big a nudge to take
for step in range(200):
    opt.zero_grad()                           # forget last step's gradients

    # ---- redo the whole forward pass (Step 3) with the CURRENT V, W ----
    W_row = F_nn.softmax(W, dim=1)
    Vw = V @ W_row.T
    logits = ((X_cat @ Vw) / 100.0).gather(1, y.unsqueeze(1)).squeeze(1)
    nll = -F_nn.logsigmoid(logits).mean()

    V_norm = F_nn.normalize(V, dim=0)
    cos_sim = (V_norm * F_nn.normalize(V_sft, dim=0)).sum(dim=0)
    reg = torch.mean(1 - cos_sim)

    loss = nll + alpha * reg
    loss.backward()   # ask PyTorch: which way should every number in V,W move to lower loss?
    opt.step()        # take that step

# =============================================================================
# STEP 6 - LOOK AT WHAT WE LEARNED  (this is where "collapse" shows up)
# =============================================================================
with torch.no_grad():
    print("final loss:", round(loss.item(), 4), " (nll:", round(nll.item(), 4),
          " reg:", round(reg.item(), 4), ")")

    # Each user's learned recipe. Watch how softmax tends to push weight onto ONE column.
    print("\nuser weights W (each row sums to 1):")
    print(F_nn.softmax(W, dim=1))

    # THE COLLAPSE CHECK: are the K basis columns pointing the same way?
    # cosine between column 0 and column 1. Near 1.0 (or -1.0) = collapsed to one direction.
    Vn = F_nn.normalize(V, dim=0)
    print("\n|cos| between the two basis columns:", round(abs((Vn[:, 0] @ Vn[:, 1]).item()), 4))
    print("(near 1.0 means the 'two' directions are really the same one -> collapse)")
