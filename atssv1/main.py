# from processors.ddp_apex_processor import DDPApexProcessor
from solver.ddp_mix_solver import DDPMixSolver
from solver.solver import SimpleSolver

# nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port 50003 main.py >>train.log 2>&1 &

if __name__ == '__main__':
    # processor = DDPMixSolver(cfg_path="config/neu.yaml")
    processor = SimpleSolver(cfg_path="config/neu.yaml")
    processor.run()
