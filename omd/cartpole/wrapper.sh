#!/bin/bash
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_ARRAY_TASK_ID	

# Parse arguments
declare -a args
declare -a values
for arg in "$@"
do
    if [[ "$arg" == *=* ]]; then
        # Argument is a value
        value=${arg#*=}
        values+=("$value")
    else
        # Argument is a flag
        if [[ ${#values[@]} -ne 0 ]]; then
            args+=("${values[@]}")
            values=()
        fi
        args+=("$arg")
    fi
done
if [[ ${#values[@]} -ne 0 ]]; then
    args+=("${values[@]}")
fi

# Loop over all combinations
function cartesian_product {
  # Adapted from https://stackoverflow.com/a/1782956
  local result=()
  local params=("${!1}")
  local lengths=()
  for param in "${params[@]}"
  do
    lengths+=($(echo "$param" | grep -o ',' | wc -l | awk '{print $1 + 1}'))
  done
  local max_index=$(( ${#params[@]} - 1 ))
  local indices=()
  for i in $(seq 0 $max_index); do indices[$i]=0; done
  local done=false
  while [[ $done == false ]]
  do
    local tuple=()
    for i in $(seq 0 $max_index)
    do
      local param="${params[$i]}"
      local index="${indices[$i]}"
      local value=$(echo "$param" | cut -d',' -f$(( $index + 1 )) --output-delimiter='')
      tuple+=("$value")
    done
    result+=("$(IFS=','; echo "${tuple[*]}")")
    local i=$max_index
    while [[ $i -ge 0 ]]
    do
      if [[ ${indices[$i]} -lt $((${lengths[$i]} - 1)) ]]
      then
        indices[$i]=$((${indices[$i]} + 1))
        for j in $(seq $(($i + 1)) $max_index)
        do
          indices[$j]=0
        done
        break
      fi
      ((i--))
    done
    if [[ $i -lt 0 ]]
    then
      done=true
    fi
  done
  echo "${result[@]}"
}

combinations=($(cartesian_product args[@]))

# export XLA_PYTHON_CLIENT_MEM_FRACTION=.80

combination=(${combinations[$SLURM_ARRAY_TASK_ID]//,/ })

metric_lr="${combination[0]}"
dim_distract="${combination[1]}"
i="${combination[2]}"
regularization_coeff="${combination[3]}"
# activation="${combination[4]}"

agent="metric"
exp=final_distract_$dim_distract

# for i in $seed $(($seed+1)); do
for activation in normalize; do
  outdir="exp/$exp/$agent/metric_lr=$metric_lr,regularization_coeff=$regularization_coeff,activation=$activation/$i"
  mkdir -p $outdir
  echo "Running $agent with arguments metric_lr=$metric_lr,regularization_coeff=$regularization_coeff,activation=$activation dim_distract=$dim_distract"
  python main.py with_inv_jac_model=True seed=$i agent_type=$agent exp=$exp dim_distract=$dim_distract metric_lr=$metric_lr \
  regularization_coeff=$regularization_coeff metric_activation=$activation out_dir=$outdir > $outdir/log.txt &
done
wait