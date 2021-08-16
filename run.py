""" A thin wrapper to call mcmc.main() that takes bash parameters
...Module author: Chen Sun
...Year: 2021
...Email: chensun@mail.tau.ac.il

"""
import sys
import getopt
import mcmc
import warnings

if __name__ == '__main__':
    # read runtime options
    warnings.filterwarnings('error', 'overflow encountered')
    warnings.filterwarnings('error', 'invalid value encountered')
    argv = sys.argv[1:]
    help_msg = 'python %s -N <number_of_steps> -o <output_folder> -L <likelihood_directory> -i <param_file> -w <number_of_walkers>' % (
        sys.argv[0])
    try:
        opts, args = getopt.getopt(argv, 'hN:o:L:i:w:')
    except getopt.GetoptError:
        raise Exception(help_msg)
    flgN = False
    flgo = False
    flgL = False
    flgi = False
    flgw = False
    for opt, arg in opts:
        if opt == '-h':
            raise Exception(help_msg)
        elif opt == '-N':
            chainslength = arg
            flgN = True
        elif opt == '-o':
            directory = arg
            flgo = True
        elif opt == '-L':
            dir_lkl = arg
            flgL = True
        elif opt == '-i':
            path_of_param = arg
            flgi = True
        elif opt == '-w':
            number_of_walkers = int(arg)
            flgw = True
    if not (flgN and flgo and flgL and flgi and flgw):
        raise Exception(help_msg)

    mcmc.main(chainslength=chainslength,
              directory=directory,
              dir_lkl=dir_lkl,
              path_of_param=path_of_param,
              number_of_walkers=number_of_walkers)
