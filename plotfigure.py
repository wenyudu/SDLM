import os
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb


job256_ld2 = ["slurm-1542683.out", "slurm-1542684.out", "slurm-1542685.out",
              "slurm-1542686.out", "slurm-1542687.out", "slurm-1542688.out",
              "slurm-1542689.out", "slurm-1542690.out", "slurm-1542691.out",
              "slurm-1542692.out", "slurm-1542693.out"]
job256_ld0 = ["slurm-1559034.out", "slurm-1559033.out", "slurm-1559032.out",
              "slurm-1559035.out", "slurm-1559038.out", "slurm-1559036.out",
              "slurm-1559031.out", "slurm-1559029.out", "slurm-1559030.out",
              "slurm-1559037.out", "slurm-1559028.out"]
job256_ld1 = ["slurm-1559022.out", "slurm-1559025.out", "slurm-1559017.out",
              "slurm-1559019.out", "slurm-1559018.out", "slurm-1559024.out",
              "slurm-1559023.out", "slurm-1559020.out", "slurm-1559027.out",
              "slurm-1559026.out", "slurm-1559021.out"]

job32_ld0 = []
job32_ld1 = []
job32_ld2 = []

job_logs = job256_ld2

logdir = "/home/hantek/scratch/syntactic_distance_lm"

log_margin = []
log_validppl = []
for filename in job_logs:
    file_handle = open(os.path.join(logdir, filename), 'r')
    exp_valid_ppl = []
    for line in file_handle.readlines():
        if "Args: Namespace" in line:
            line = line.split(',')
            for token in line:
                if "margin=" in token:
                    margin_value = token.split('=')[1]
                    log_margin.append(margin_value)
        if "valid ppl" in line and "| end of epoch" in line:
            line = line.split('|')
            for token in line:
                if "valid ppl" in token:
                    valid_ppl = [xx for xx in token.split(' ') if xx]
                    valid_ppl = float(valid_ppl[2])
                    exp_valid_ppl.append(valid_ppl)
    log_validppl.append(exp_valid_ppl)


assert len(log_margin) == len(log_validppl)

for margin, valid_curve in zip(log_margin, log_validppl):
    color = 1.0 * (-numpy.log(float(margin))) / len(log_margin)
    plt.plot([i for i, j in enumerate(valid_curve)],
             valid_curve,
             color=(0, color, 0, 1),
             label=margin)

plt.title("PPL for different margins")
plt.ylabel('PPL')
plt.ylim(64, 70)
plt.xlabel('Number of epochs')
plt.legend()

plt.savefig("batchsize256.png")

pdb.set_trace()
