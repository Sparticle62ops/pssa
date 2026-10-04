# Local, optional adapter; no downloads. Kept out of the training/data CLI path.
import json
import sys

try:
    import pyarrow.parquet as pq
    source = pq.ParquetFile(sys.argv[1])
    names = source.schema_arrow.names
    field = next((n for n in ("text", "content", "body") if n in names), None)
    if field is None:
        raise ValueError("Parquet needs a text, content or body column")
    print(json.dumps({"records": source.metadata.num_rows}), flush=True)
    for batch in source.iter_batches(batch_size=64, columns=[field], use_threads=False):
        for value in batch.column(0).to_pylist():
            if value is not None and not isinstance(value, str):
                raise ValueError("selected Parquet column must contain strings")
            if value is not None and len(value) > 1048576:
                raise ValueError("record exceeds 1 MiB; split long records first")
            print(json.dumps(value or "", ensure_ascii=True), flush=True)
except Exception as error:
    print(json.dumps({"error": "Local Parquet reader: " + str(error) + "; requires python3 + pyarrow"}), flush=True)
    sys.exit(1)
