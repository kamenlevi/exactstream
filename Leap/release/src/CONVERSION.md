# Conversion

The expert store is built in three steps from the model's safetensors: `convert/repack_records.py` (the expert pieces,
one record per expert), `convert/make_records3.py` (each 4-bit code split into a 3-bit part and a 1-bit part, with a
verifier that checks every expert from disk), and `convert/convert_records_p11.py --pad` (the native piece order and
11-bit packed scales the kernels read).

    bash convert/convert_store.sh /path/to/model-dir /path/to/model-dir/records3

runs the three steps one layer at a time: about 3-4.4 GB of extra disk at a time beside the growing 68.5 GB store, and a
stopped run resumes at the next unfinished layer. `convert/test_layers.sh` runs them on a few layers and compares the
bytes with an existing store. On the test MacBook Pro a full conversion of all 48 layers took 9.5 minutes, and the store it built
equals, byte for byte, the one all our measurements used.
