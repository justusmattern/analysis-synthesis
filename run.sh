#!/bin/bash
for i in {1..30}
do
   python eval_ppl.py --seed 32 --model_type ae_vae_fusion --dataset wi --add_attn --model-path ./out/civilcomments_unfrozen/model_aproved_unfrozen_epoch$i.pt
done

