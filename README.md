# ULDM_x_SPARC

This is a code to constrain the ultralight dark matter (ULDM) using the SPARC data set. The running modes contain {smart grid, even grid} x {soliton + NFW, soliton + Burkert, soliton only}. For soliton + NFW/Burkert, to speed up the marginalization of nuisance parameters, we recommend using `emcee` to propose a smart grid. 

## Usage

Simple use case of a run with smart-grid can be invoked by

```bash
python run.py -N <length_of_chain> -o <path_to_output> -L <location_of_dataset> -i <path_to_input> -w <number_of_walkers>
```

for example:

```bash
python run.py -N 30000 -o chains/run_18_ma_24 -L ./data/ -i input/sample.param -w 100
```

where `sample.param` is the param card to be specified separately, with the range of the scan as well as galaxies of your choice. 

After the run finishes, the chains can be either parsed by

```bash
python analyze.py -i <path_to_chain>
```
or by using the `demo_parse_smart_grid.ipynb`.

The m slice run can also be automated with run_mslicing.py:
```bash
python run_mslicing.py -N <length_of_chain> -o <path_to_output> -L <location_of_dataset> -i <path_to_input> -w <number_of_walkers> -m 'logm_min logm_max number_of_slicing' -G 'galA galB ...'
```

For a scan of Model C with evenly spaced grid, see `demo_even_grid_scan.ipynb`. 

## License

This code is under [MIT](https://opensource.org/licenses/MIT) license. 

## Bibtex entry
If you use this code or find it in any way useful for your research, please cite Bar, Blum, and Sun (2021). The Bibtex entry is: *tba*


