#!/bin/bash
# MD5SUMS of the build's shipped files (Fable FAR): the scripts, the patch and the patched files, the base copies they
# were made against, and every mask. install_nw.sh refuses to install a folder that differs from it. Logs, INSTALL.md
# and the scratch / check folders are not listed. Usage: bash mk_md5.sh   (rewrites MD5SUMS)
cd "$(dirname "$0")" || exit 1
md5 -r nw_patch.py test_nw.py check_install.py install_nw.sh nw_masks.py mk_builder.py check_masks.py nw_selfcheck.py nw_read.py \
  neverwait.patch nw_masks.diff fri_extra_nw.txt base/*.py patched/*.py masks/*.stats.json masks/*/*.npz > MD5SUMS
echo "MD5SUMS: $(wc -l < MD5SUMS | tr -d ' ') files"
