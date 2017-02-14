import argparse
import glob
import os


# sample batch job: qsub -l nodes=4:ppn=12,mem=88gb,walltime=24:00:00 parallel_diffuse.pbs

def preamble(job_name, num_jobs, mem=4000, walltime='24:00:00', theano=False):
    pre = """#!/bin/sh
# Script for running serial program, diffuse.
#PBS -l walltime=%s
#PBS -l mem=%dmb
#PBS -r n
#PBS -q gpu
#PBS -j eo
#PBS -N %s
#PBS -t 1-%d

""" % (walltime, mem, job_name, num_jobs)
    if theano:
        return pre + """
module load intel/12
module load python
module load cuda/7.5.18

cd /home/anuar12/deep_bgt/game_net/
export THEANO_FLAGS="floatX=float32,device=gpu,mode='FAST_RUN',force_device=True,base_compiledir=/global/scratch/anuar12/.theano/%s-$PBS_ARRAYID"
export PYTHONPATH=$PYTHONPATH:/home/anuar12/deep_bgt/game_net/

echo "Current working directory is `pwd`"
echo "Starting run at: `date`"
""" % job_name
    else:
        return pre

def parse_args():
    parser = argparse.ArgumentParser(description="Job file builder script")
    parser.add_argument('-f', default=None,
        help="Path of json file describing the options of the experiment")
    parser.add_argument('--cersei', dest='cersei', action='store_true')
    parser.set_defaults(cersei=False)
    return parser.parse_args()

def addendum(job_name):
    return """
\\rm -r /global/scratch/anuar12/.theano/%s-$PBS_ARRAYID

""" % (job_name)

# Write batch job script for Westgrid
def build_job_file(joblist, filename, mem=8000, walltime='36:00:00', theano=False):
    with(open(filename+'.pbs', 'w')) as f:
        f.write(preamble(filename, len(joblist), mem, walltime, theano=theano))
        f.write('case $PBS_ARRAYID in\n')
        for i, job in enumerate(joblist):
            f.write('%d) %s --start_fold %d --end_fold %d;;\n' % (i + 1, job, i, i + 1))
        f.write('''*) echo "Unrecognized jobid '$PBS_ARRAYID'"\n''')
        f.write('esac\n')
        # if theano:
        #     f.write(addendum(filename))

def build_local_job_file(joblist, filename):
    with(open(filename+'.sh', 'w')) as f:
        for i, job in enumerate(joblist):
            out = 'THEANO_FLAGS=device=cpu %s' % (job)
            if i != 7:
                out += ' &' 
            f.write(out+'\n')

def main():
    args = parse_args()
    if args.f is None:
        joblist = ['python test.py --seed %d' % i for i in range(100)]
        filename = 'TEST'
        theano = False
    else:
        files = glob.glob(args.f+'*.json')
        files.sort()
        joblist = ['python kfold.py --json %s' % os.path.abspath(f) for f in files]
        filename = args.f[0:-1]
        theano = True
    if not args.cersei:
        build_job_file(joblist, filename=filename, theano=theano)
    else:
        build_local_job_file(joblist, filename=filename)


# Usage: -f "path_to_dir_of_json_settings"
# Generates a .pbs batch job script for WestGrid
if __name__=="__main__":
    main()
