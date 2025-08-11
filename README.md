This is a ML pipeline for a neutrino flavor stability classifier.


File Organization:

ML_proj/
└─ model_training/
   ├─ configs/
   │  ├─ config.yaml
   │  ├─ data/
   │  │  └─ mydataset.yaml
   │  ├─ model/
   │  │  └─ auto_model.yaml
   │  └─ train/
   │     └─ base.yaml
   ├─ data/                 
   │  ├─ raw/               
   │  ├─ interim/           
   │  └─ processed/         
   ├─ artifacts/            
   ├─ scripts/
   │  ├─ slurm/
   │  │  ├─ train.sbatch
   │  │  └─ env.sh          
   │  └─ docker/            
   ├─ tests/                >>>>>> This doesn't exis in full yet
   │  ├─ test_model_forward.py
   │  ├─ test_dataset.py
   │  └─ test_smoke_loop.py
   └─ src/
      └─ rhea_train/
         ├─ __init__.py             
         ├─ train.py           
         ├─ predict.py
         ├─ data/
         │  ├─ __init__.py
         │  ├─ registry.py
         │  ├─ datasets.py
         │  ├─ datamodules.py   
         │  ├─ transforms.py
         │  └─ generators/
         │     └─ example_gaussians.py
         ├─ models/
         │  ├─ __init__.py
         │  ├─ registry.py
         │  ├─ nn_blocks.py
         │  └─ <your_model>.py
         ├─ losses/
         │  └─ <task>.py
         ├─ metrics/
         │  └─ <task>.py
         ├─ engine/
         │  ├─ loops.py         
         │  ├─ callbacks.py     
         │  ├─ optim.py         
         │  └─ distributed.py  
         ├─ inference/
         │  ├─ export.py       >>>>> C++ export
         └─ utils/
            ├─ io.py
            ├─ logging.py       
            ├─ randomness.py  >>>> seeding/determinism
            └─ config.py        
