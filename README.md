# Graph Wiring for Dorothea dataset

What is Graph Wiring? Any vector space is also a graph and `arrowspace` builds and leverage this graph -> [blog post](https://www.tuned.org.uk/posts/010_game_changer_unifying_vectors_and_features_graphs)

## Usage

These scripts need 35Gbs of RAM to run. You can still run the "00" (even if it fails with out of memory still produce the parquet file), "01", "02" and "03" scripts and check the analysis.

* Download and unzip `https://archive.ics.uci.edu/static/public/169/dorothea.zip` in the `data` directory
* `python 00_ingestion.py --data-dir data/DOROTHEA/` (this may fail with "process killed" but only the generated ~900Mb parquet file is needed to proceed)
* `python 02_preliminary.py` (analyse the raw data)
* `python 03_estimate_params_wiring.py` (estimate best parameters to run the wiring)
* `cd dorothea_wiring && cargo run --release` (Rust version is needed as the Python version take ~25% more memory)


## Testing

Detach: `byobu new-session -d -s "my-session" "RUST_LOG=info cargo run --release > output.txt 2>&1"`