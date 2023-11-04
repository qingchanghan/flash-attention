

# nsight compute
ncu \
        --target-processes all \
        --export report_transpose_1024_v4_v5 \
        --import-source=yes \
        --page raw \
        --set full \
        --force-overwrite \
        python test_transpose.py
