from astropy.table import Table
import astropy.units as u
from astroquery.gaia import Gaia
import numpy as np
import time

data = np.load(
    "pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.npz"
)["data"]

source_id = list(set(data["sourceID"]))

table = Table([source_id], names=["source_id"])

Gaia.login()
Gaia.upload_table(upload_resource=table, table_name='gcns_wds')

result = Gaia.launch_job(
    "SELECT top 20000 g.* from gaiaedr3.gaia_source as g, user_mlam01.gcns_wds as u where g.source_id = u.source_id",
    verbose=True,
).get_results()


np.savez_compressed(
    "pubgcnswdlf-crossmatched-full-edr3-catalogue", data=result
)
