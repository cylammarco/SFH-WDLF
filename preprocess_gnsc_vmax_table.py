import numpy as np

# dtype=[
#     ('Mbol', '<f2'),
#     ('VoverVmax', '<f4'),
#     ('Vmax', '<f4'),
#     ('VoverVgen', '<f4'),
#     ('Vgen', '<f4'),
#     ('sourceID', '<i8'),
#     ('ra', '<f4'),
#     ('dec', '<f4'),
#     ('dpc', '<f2')
# ])
data = np.genfromtxt(
    "pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.csv",
    delimiter=",",
    names=True,
    dtype=[
        "float32",
        "float32",
        "float64",
        "float32",
        "float64",
        "int64",
        "float32",
        "float32",
        "float16",
    ],
)

np.savez_compressed(
    "pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt", data=data
)
