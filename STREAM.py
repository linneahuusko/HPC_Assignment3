from timeit import default_timer as timer
import numpy as np
from sizeof import sizeof
from array import array
import matplotlib.pyplot as plt
import cythonSTREAM

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

times = np.zeros(4)

arraytypes = ["array", "numpy"]
offset = 0.25

sizes = [10, 100, 10000]

for ax, STREAM_ARRAY_SIZE in zip(axes, sizes):
    for i, arraytype in enumerate(arraytypes):
        # Initialize arrays
        if arraytype == "array":
            a = array("d", range(STREAM_ARRAY_SIZE))
            b = array("d", range(STREAM_ARRAY_SIZE))
            c = array("d", range(STREAM_ARRAY_SIZE))

        elif arraytype == "numpy":
            a = np.array(range(STREAM_ARRAY_SIZE), dtype=np.double)
            b = np.array(range(STREAM_ARRAY_SIZE), dtype=np.double)
            c = np.array(range(STREAM_ARRAY_SIZE), dtype=np.double)
        STREAM_ARRAY_TYPE = type(a)

        # Set initial values in the arrays
        for j in range(STREAM_ARRAY_SIZE):
            a[j] = 1.0
            b[j] = 2.0
            c[j] = 0.0

        times = cythonSTREAM.STREAM(a, b, c, times)

        # Compute the bandwidth
        data_amount = np.zeros(4)
        data_amount[0] = 2 * sizeof(STREAM_ARRAY_TYPE) * STREAM_ARRAY_SIZE
        data_amount[1] = 2 * sizeof(STREAM_ARRAY_TYPE) * STREAM_ARRAY_SIZE
        data_amount[2] = 3 * sizeof(STREAM_ARRAY_TYPE) * STREAM_ARRAY_SIZE
        data_amount[3] = 3 * sizeof(STREAM_ARRAY_TYPE) * STREAM_ARRAY_SIZE

        bandwidth = data_amount / times

        # Plot results
        ax.bar(
            np.arange(0, 4) + i * offset + 0.125, bandwidth / 1e9, label=arraytype, width=0.25
        )

    ax.set_xticks([0.25, 1.25, 2.25, 3.25])
    ax.set_xticklabels(["Copy", "Scale", "Sum", "Triad"])
    ax.set_title(f"Array size: {STREAM_ARRAY_SIZE}")
axes[0].legend()
axes[0].set_ylabel("Bandwidth [GB/s]")
fig.suptitle("Cythonized", y=1.05, fontsize=16)
fig.subplots_adjust(wspace=0.1)
plt.savefig("Exercise1.png", bbox_inches="tight")
