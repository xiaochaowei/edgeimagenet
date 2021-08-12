ngc batch run --name "ml-model.imagenet 21_448_wd0.1_lr8e-4" --preempt RUNONCE \
 --total-runtime 86400s \
  --image "nvcr.io/nvidia/pytorch:20.12-py3" \
  --ace nv-us-west-2 --instance dgx1v.32g.4.norm \
  --result /result --org nvidian --datasetid 13254:/dataset \
  --workspace ws-chaowei1:/workspace \
  --commandline "bash run.sh'"