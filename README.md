# ULDM_x_SPARC

This is a code to constrain the ultralight dark matter (ULDM) using the SPARC data set. The running modes contain {smart grid, even grid} x {soliton + NFW, soliton + Burkert, soliton only}. For soliton + NFW/Burkert, to speed up the marginalization of nuisance parameters, I recommend using `emcee` to propose a smart grid as it is defaulted now.

## Results
<p align="center">
  <img src="./notebooks/plots/sol_full_SPARC_2sigma.png" width="500" alt="soliton mass contained in each disk galaxy" title="soliton mass contained in each disk galaxy">
  <img src="./notebooks/plots/f_2sigma.png" width="500" alt="ULDM total density fraction" title="ULDM total density fraction">
</p>

First plot is the constraint on soliton mass contained in each disk galaxy (normalized by the amount predicted by simulations,) second interpreted constraint on ULDM total density. See the publication for details.

## Usage

Simple use case of a run with smart-grid can be invoked by

```bash
python run.py -N <length_of_chain> -o <path_to_output> -L <location_of_dataset> -i <path_to_input> -w <number_of_walkers>
```

for example:

```bash
python run.py -N 30000 -o ./chains/run_18_ma_24 -L ./data/ -i input/sample.param -w 100
```

where `sample.param` is the param card to be specified separately, with the range of the scan as well as galaxies of your choice. 

After the run finishes, the chains can be either parsed by

```bash
python analyze.py -i <path_to_chain>
```
or by using the `1_demo_parse_smart_grid.ipynb`.

The m slice run can also be automated with run_mslicing.py:
```bash
python run_mslicing.py -N <length_of_chain> -o <path_to_output> -L <location_of_dataset> -i <path_to_input> -w <number_of_walkers> -m 'logm_min logm_max number_of_slicing' -G 'galA galB ...'
```

for example:
```bash
python run_mslicing.py -N 30000 -o ./chains/run_000_NFW -L ./data -i input/sample_mslicing_2.param -w 100 -m '-25 -19 30' -G 'NGC0100 NGC2403'
```

will automatically run soliton+NFW model over NGC0100 and NGC2403 with 30 fixed m ranging from 1e-25 eV to 1e-19 eV. The h5 chains will be saved under ./chains/ folder. 

For a scan of Soliton+anything with evenly spaced grid, see `2_demo_even_grid_scan.ipynb`. For discussions on gravitational dynamical relaxation time scales, see `3_demo_dynamical_relaxation_time.ipynb`.

## License

This code is under [MIT](https://opensource.org/licenses/MIT) license. 

## BibTeX entry
If you use this code or find it in any way useful for your research, please cite Bar, Blum, and Sun (2021). The BibTeX entry is: 

    @article{Bar:2021kti,
        author = "Bar, Nitsan and Blum, Kfir and Sun, Chen",
        title = "{Galactic rotation curves vs. ultralight dark matter II}",
        eprint = "2111.03070",
        archivePrefix = "arXiv",
        primaryClass = "hep-ph",
        month = "11",
        year = "2021"
    }


