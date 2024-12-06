import multiprocessing
import pathlib

from datetime import datetime

from tobac_flow.linking import find_overlap_between_files, process_linking_output

import argparse
parser = argparse.ArgumentParser(
    description="Combine multiple files of detected DCCs in GOES-16 ABI data"
)
parser.add_argument("files", help="List of files to combine", nargs="+", type=str)
parser.add_argument("save_path", help="Path to save results of label linking", type=str)
parser.add_argument("-n", help="Number of cores", default=1, type=int)

args = parser.parse_args()

files = [pathlib.Path(f).resolve() for f in args.files]
save_path = pathlib.Path(args.save_path).resolve()
n_cores = int(args.n)

if __name__ == '__main__':
    start = datetime.now()
    print(f'Linking {len(files)} files', flush=True)
    print(f'Commencing linking using {n_cores} processes')
    with multiprocessing.Pool(n_cores) as p:
        overlap_results = p.starmap(find_overlap_between_files, zip(files, files[1:]))
        p.close()
        p.join()
    
    print("Processing linking output", flush=True)
    dataset = process_linking_output(overlap_results)

    print(f'Saving output to {save_path}', flush=True)
    dataset.to_netcdf(save_path)

    print("Linking finished", datetime.now())
    print("Time elapsed:", datetime.now() - start)