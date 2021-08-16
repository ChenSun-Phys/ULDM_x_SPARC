# ULDM_x_SPARC

This is a code to constrain the ultralight dark matter using SPARC galaxies. The running modes contain soliton + NFW, soliton + Burkert, and soliton only. For the first two cases, to speed up the marginalization of nuisance parameters, we use `emcee` to propose a smart grid. 

## Usage

Simple use case of a run with smart-grid can be invoked by

```bash
python python_code/run.py -N <length_of_chain> -o <path_to_output> -L <location_of_dataset> -i <path_to_input> -w <number_of_walkers>
```

for example:

```bash
python python_code/run.py -N 30000 -o chains/run_18_ma_24 -L ./data/ -i input/sample4.param -w 100
```

After the run finishes, the chains can be either parsed by

```bash
python python_code/analyze.py -i <path_to_chain>
```
or by using the `demo_parse_smart_grid.ipynb`.

The m slice run can also be automated with run_mslicing.py:
```bash
python python_code/run_mslicing.py -N <length_of_chain> -o <path_to_output> -L <location_of_dataset> -i <path_to_input> -w <number_of_walkers> -m 'logm_min logm_max number_of_slicing' -G 'galA galB ...'
```

For a scan of Model C with evenly spaced grid, see `demo_even_grid_scan.ipynb`. 

## License

This code is under [MIT](https://opensource.org/licenses/MIT) license. 

## Bibtex entry
If you use this code or find it in any way useful for your research, please cite Bar, Blum, and Sun (2021). The Bibtex entry is: *tba*


